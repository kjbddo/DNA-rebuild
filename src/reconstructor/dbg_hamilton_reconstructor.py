import time
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from typing import List, Dict, Set, Tuple
import numpy as np
from collections import defaultdict
from src.generator.generate_dna_sequence import DNASequence
from src.sequence_comparison import compare_sequences
from src.sequence_reconstruction import (
    read_reads_streaming,
    save_reconstructed_sequence
)

class DBGHamiltonReconstructor:
    def __init__(self, k: int, read_length, max_mismatches, chunk_size):
        self.k = k
        self.read_length = read_length
        self.max_mismatches = max_mismatches
        self.chunk_size = chunk_size
        self.base_map = {0: 'A', 1: 'T', 2: 'C', 3: 'G'}
        self.base_reverse_map = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
        
    def create_kmers(self, read: np.ndarray) -> List[str]:
        """리드에서 k-mer 생성"""
        kmers = []
        for i in range(len(read) - self.k + 1):
            kmer = ''.join(self.base_map[b] for b in read[i:i+self.k])
            kmers.append(kmer)
        return kmers

    def build_hamilton_graph(self, reads: List[np.ndarray]) -> Tuple[Dict, Dict]:
        """해밀턴 그래프 구축"""
        print("\n=== 해밀턴 그래프 구축 시작 ===")
        
        # 그래프의 간선과 가중치를 저장할 딕셔너리
        edges = defaultdict(set)  # kmer -> set of next kmers
        weights = defaultdict(int)  # (kmer1, kmer2) -> weight
        
        # 각 리드에서 k-mer를 추출하여 그래프 구축
        total_reads = len(reads)
        for i, read in enumerate(reads):
            kmers = self.create_kmers(read)
            for j in range(len(kmers) - 1):
                kmer1, kmer2 = kmers[j], kmers[j+1]
                edges[kmer1].add(kmer2)
                weights[(kmer1, kmer2)] += 1
                
            if (i + 1) % 100 == 0:
                print(f"\r진행률: {((i+1)/total_reads)*100:.1f}% ({i+1}/{total_reads} 리드)", end="")
                
        print("\n\n=== 그래프 구축 완료 ===")
        print(f"노드 수: {len(edges):,}")
        print(f"간선 수: {sum(len(suffixes) for suffixes in edges.values()):,}")
        
        return edges, weights

    def find_hamilton_path(self, edges: Dict, weights: Dict) -> List[str]:
        """해밀턴 경로 찾기 (미스매치 개수 기준)"""
        if not edges:
            return []
            
        start_node = max(edges.keys(), key=lambda x: len(edges[x]))
        path = [start_node]
        visited = {start_node}
        
        while True:
            current = path[-1]
            if current not in edges:
                break
                
            next_node = None
            max_weight = self.min_coverage
            
            for neighbor in edges[current]:
                weight = weights[(current, neighbor)]
                if neighbor not in visited and weight > max_weight:
                    # 미스매치 개수 계산
                    mismatches = sum(1 for i in range(self.k-1) 
                                   if current[i+1] != neighbor[i])
                    if mismatches <= self.max_mismatches:
                        next_node = neighbor
                        max_weight = weight
            
            if next_node is None:
                break
                
            path.append(next_node)
            visited.add(next_node)
            
        return path

    def path_to_sequence(self, path: List[str]) -> np.ndarray:
        """해밀턴 경로를 DNA 시퀀스로 변환"""
        if not path:
            return np.array([], dtype=np.uint8)
            
        # 첫 번째 k-mer는 전체 포함
        sequence = list(path[0])
        
        # 나머지는 마지막 문자만 추가
        for kmer in path[1:]:
            sequence.append(kmer[-1])
            
        # 문자열을 숫자 배열로 변환
        return np.array([self.base_reverse_map[base] for base in sequence], 
                       dtype=np.uint8)

    def reconstruct(self, reads_file: str) -> np.ndarray:
        """해밀턴 경로 기반 시퀀스 재구성"""
        print("\n=== 해밀턴 경로 기반 시퀀스 재구성 시작 ===")
        
        # 리드 데이터 읽기
        reads_list = []
        for chunk in read_reads_streaming(reads_file, self.read_length, self.chunk_size):
            reads_list.extend(chunk)
            
        if not reads_list:
            print("리드 데이터를 읽을 수 없습니다.")
            return np.array([], dtype=np.uint8)
            
        # 해밀턴 그래프 구축
        edges, weights = self.build_hamilton_graph(reads_list)
        
        # 해밀턴 경로 찾기
        print("\n=== 해밀턴 경로 탐색 시작 ===")
        path = self.find_hamilton_path(edges, weights)
        
        # 경로를 시퀀스로 변환
        reconstructed = self.path_to_sequence(path)
        print(f"\n재구성된 시퀀스 길이: {len(reconstructed)}bp")
        
        # 재구성된 시퀀스 저장
        bin_path, txt_path = save_reconstructed_sequence(reconstructed, reads_file, "dbg-hamilton")
        
        return bin_path, txt_path

if __name__ == "__main__":
    k = 31
    read_length = 100
    max_mismatches = 2
    chunk_size = 10000
    overlap = 50
    
    generator = DNASequence()
    bin_path, txt_path = generator.save_sequence(chunk_size, "test.bin")
    reads_bin_path, reads_txt_path = generator.save_reads(bin_path, read_length, overlap, chunk_size)
    
    reconstructor = DBGHamiltonReconstructor(k, read_length, max_mismatches, chunk_size)
    
    start_time = time.time()
    reconstructed_bin_path, reconstructed_txt_path = reconstructor.reconstruct(reads_bin_path)
    end_time = time.time()
    
    print(f"\n실행 시간: {end_time - start_time:.2f}초")
    
    original_file = bin_path
    reconstructed_file = reconstructed_bin_path
    compare_sequences(original_file, reconstructed_file)

