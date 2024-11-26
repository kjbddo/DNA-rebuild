import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.generator.generate_dna_sequence import DNASequence
from typing import List, Dict, Optional
import numpy as np
from src.sequence_reconstruction import (
    read_reads_streaming,
    save_reconstructed_sequence
)

class BruteForceReconstructor:
    def __init__(self, k: int = 31, min_coverage: int = 2, read_length: int = 100):
        self.k = k
        self.min_coverage = min_coverage
        self.read_length = read_length
        self.base_map = {0: 'A', 1: 'T', 2: 'C', 3: 'G', 4: 'N'}
        self.base_reverse_map = {'A': 0, 'T': 1, 'C': 2, 'G': 3, 'N': 4}

    def find_best_overlap(self, read1: np.ndarray, read2: np.ndarray, min_overlap: int = 10) -> int:
        """두 리드 간의 최대 중첩 길이를 찾는 함수"""
        max_overlap = 0
        read1_len = len(read1)
        read2_len = len(read2)
        
        for overlap in range(min(read1_len, read2_len), min_overlap-1, -1):
            if np.array_equal(read1[-overlap:], read2[:overlap]):
                max_overlap = overlap
                break
                
        return max_overlap

    def reconstruct(self, reads_file: str) -> np.ndarray:
        """브루트 포스 방식으로 시퀀스 재구성"""
        print("\n=== 브루트 포스 시퀀스 재구성 시작 ===")
        
        # 리드 데이터 읽기
        reads_list = []
        for chunk in read_reads_streaming(reads_file, self.read_length):
            reads_list.extend(chunk)
            
        if not reads_list:
            print("리드 데이터를 읽을 수 없습니다.")
            return np.array([], dtype=np.uint8)
            
        # 첫 번째 리드로 시작
        reconstructed = reads_list[0].copy()
        used_reads = {0}
        
        while len(used_reads) < len(reads_list):
            best_overlap = 0
            best_read_idx = -1
            is_prefix = True
            
            # 남은 리드들 중 가장 좋은 중첩을 찾음
            for i in range(len(reads_list)):
                if i in used_reads:
                    continue
                    
                # 접미사 중첩 확인
                suffix_overlap = self.find_best_overlap(reconstructed, reads_list[i])
                if suffix_overlap > best_overlap:
                    best_overlap = suffix_overlap
                    best_read_idx = i
                    is_prefix = False
                    
                # 접두사 중첩 확인
                prefix_overlap = self.find_best_overlap(reads_list[i], reconstructed)
                if prefix_overlap > best_overlap:
                    best_overlap = prefix_overlap
                    best_read_idx = i
                    is_prefix = True
            
            if best_read_idx == -1:
                print("최적의 리드를 찾을 수 없습니다.")
                break
                
            # 선택된 리드를 재구성된 시퀀스에 추가
            best_read = reads_list[best_read_idx]
            if is_prefix:
                reconstructed = np.concatenate([best_read[:-best_overlap], reconstructed])
            else:
                reconstructed = np.concatenate([reconstructed, best_read[best_overlap:]])
                
            used_reads.add(best_read_idx)
            
            if len(used_reads) % 10 == 0:
                print(f"\r진행률: {(len(used_reads)/len(reads_list))*100:.1f}% ({len(used_reads)}/{len(reads_list)} 리드)", end="")
                
        print(f"\n\n재구성된 시퀀스 길이: {len(reconstructed)}bp")
        
        # 재구성된 시퀀스 저장
        save_reconstructed_sequence(reconstructed, reads_file, "brute-force")
        
        return reconstructed

if __name__ == "__main__":
    generator = DNASequence()
    bin_path, txt_path = generator.save_sequence(1000, "test.bin")
    bin_path, txt_path = generator.save_reads(bin_path, 100, 50, 1000)
    reconstructor = BruteForceReconstructor()
    reconstructor.reconstruct("data/reads/sequence_1000bp_reads_100bp.bin")
