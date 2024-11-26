import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.sequence_comparison import compare_sequences
from src.generator.generate_dna_sequence import DNASequence
from typing import List, Dict, Set, Tuple
import numpy as np
from collections import defaultdict, deque
from src.sequence_reconstruction import (
    read_reads_streaming,
    save_reconstructed_sequence
)

class DeBruijnReconstructor:
    def __init__(self, k: int = 31, min_coverage: int = 2, read_length: int = 100):
        self.k = k
        self.min_coverage = min_coverage
        self.read_length = read_length
        self.base_map = {0: 'A', 1: 'T', 2: 'C', 3: 'G', 4: 'N'}
        self.base_reverse_map = {'A': 0, 'T': 1, 'C': 2, 'G': 3, 'N': 4}
        
    def create_kmers(self, read: np.ndarray, k: int) -> List[Tuple[str, str]]:
        """리드에서 k-mer와 (k-1)-mer 쌍을 생성"""
        kmers = []
        for i in range(len(read) - k + 1):
            kmer = ''.join(self.base_map[b] for b in read[i:i+k])
            prefix = kmer[:-1]  # (k-1)-mer prefix
            suffix = kmer[1:]   # (k-1)-mer suffix
            kmers.append((prefix, suffix))
        return kmers

    def build_debruijn_graph(self, reads: List[np.ndarray]) -> Tuple[Dict, Dict]:
        """De Bruijn 그래프 구축"""
        print("\n=== De Bruijn 그래프 구축 시작 ===")
        
        # 그래프의 간선과 가중치를 저장할 딕셔너리
        edges = defaultdict(set)  # prefix -> set of suffixes
        weights = defaultdict(int) # (prefix, suffix) -> weight
        
        # 각 리드에서 k-mer를 추출하여 그래프 구축
        total_reads = len(reads)
        for i, read in enumerate(reads):
            kmers = self.create_kmers(read, self.k)
            for prefix, suffix in kmers:
                edges[prefix].add(suffix)
                weights[(prefix, suffix)] += 1
                
            if (i + 1) % 100 == 0:
                print(f"\r진행률: {((i+1)/total_reads)*100:.1f}% ({i+1}/{total_reads} 리드)", end="")
                
        print("\n\n=== 그래프 구축 완료 ===")
        print(f"노드 수: {len(edges):,}")
        print(f"간선 수: {sum(len(suffixes) for suffixes in edges.values()):,}")
        
        return edges, weights

    def find_eulerian_path(self, edges: Dict, weights: Dict) -> List[str]:
        """De Bruijn 그래프에서 오일러 경로 찾기"""
        if not edges:
            return []
            
        # 시작 노드 선택 (들어오는 간선보다 나가는 간선이 많은 노드)
        in_degree = defaultdict(int)
        out_degree = defaultdict(int)
        
        for node, suffixes in edges.items():
            out_degree[node] = len(suffixes)
            for suffix in suffixes:
                in_degree[suffix] += 1
                
        start_node = next(iter(edges.keys()))  # 기본값
        for node in edges:
            if out_degree[node] - in_degree[node] == 1:
                start_node = node
                break
                
        # Hierholzer's 알고리즘으로 오일러 경로 찾기
        path = []
        stack = [start_node]
        current_path = []
        
        while stack:
            current_node = stack[-1]
            
            if current_node in edges and edges[current_node]:
                next_node = min(edges[current_node], 
                              key=lambda x: weights[(current_node, x)])
                edges[current_node].remove(next_node)
                stack.append(next_node)
            else:
                current_path.append(stack.pop())
                
        return current_path[::-1]  # 역순으로 반환

    def path_to_sequence(self, path: List[str]) -> np.ndarray:
        """오일러 경로를 DNA 시퀀스로 변환"""
        if not path:
            return np.array([], dtype=np.uint8)
            
        # 첫 번째 (k-1)-mer는 전체 포함
        sequence = list(path[0])
        
        # 나머지는 마지막 문자만 추가
        for node in path[1:]:
            sequence.append(node[-1])
            
        # 문자열을 숫자 배열로 변환
        return np.array([self.base_reverse_map[base] for base in sequence], 
                       dtype=np.uint8)

    def reconstruct(self, reads_file: str) -> np.ndarray:
        """De Bruijn 그래프를 사용한 시퀀스 재구성"""
        print("\n=== De Bruijn 시퀀스 재구성 시작 ===")
        
        # 리드 데이터 읽기
        reads_list = []
        for chunk in read_reads_streaming(reads_file, self.read_length):
            reads_list.extend(chunk)
            
        if not reads_list:
            print("리드 데이터를 읽을 수 없습니다.")
            return np.array([], dtype=np.uint8)
            
        # De Bruijn 그래프 구축
        edges, weights = self.build_debruijn_graph(reads_list)
        
        # 오일러 경로 찾기
        print("\n=== 오일러 경로 탐색 시작 ===")
        path = self.find_eulerian_path(edges, weights)
        
        # 경로를 시퀀스로 변환
        reconstructed = self.path_to_sequence(path)
        print(f"\n재구성된 시퀀스 길이: {len(reconstructed)}bp")
        
        # 재구성된 시퀀스 저장
        save_reconstructed_sequence(reconstructed, reads_file, "debruijn")
        
        return reconstructed

if __name__ == "__main__":
    generator = DNASequence()
    bin_path, txt_path = generator.save_sequence(10**8, "test.bin")
    bin_path, txt_path = generator.save_reads(bin_path, 100, 41, 10**9)
    
    reconstructor = DeBruijnReconstructor()
    
    print("De Bruijn 시퀀스 재구성 시작")
    start_time = time.time()
    reconstructor.reconstruct(bin_path)
    end_time = time.time()
    
    print(f"\n실행 시간: {end_time - start_time:.2f}초")
    
    original_file = "data/original/sequence_10000bp.bin"
    reconstructed_file = "data/reconstructed/debruijn_sequence_10000bp_reads_100bp.bin"
    compare_sequences(original_file, reconstructed_file)

