import sys
import os
import time


sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.sequence_comparison import compare_sequences
from src.generator.generate_dna_sequence import DNASequence
from typing import List, Dict, Optional
import numpy as np
from src.sequence_reconstruction import (
    read_reads_streaming,
    save_reconstructed_sequence
)

class BruteForceReconstructor:
    def __init__(self, min_coverage: int = 2, read_length: int = 100, max_mismatches: int = 2):
        self.min_coverage = min_coverage
        self.read_length = read_length
        self.max_mismatches = max_mismatches  # 허용할 최대 미스매치 수
        self.base_map = {0: 'A', 1: 'T', 2: 'C', 3: 'G'}
        self.base_reverse_map = {'A': 0, 'T': 1, 'C': 2, 'G': 3}

    def find_best_overlap(self, read1: np.ndarray, read2: np.ndarray, min_overlap: int = 10) -> tuple:
        """두 리드 간의 최대 중첩 길이와 불일치 수를 찾는 함수"""
        max_overlap = 0
        min_mismatches = float('inf')
        best_position = 0
        read1_len = len(read1)
        read2_len = len(read2)
        
        for overlap in range(min(read1_len, read2_len), min_overlap-1, -1):
            mismatches = np.sum(read1[-overlap:] != read2[:overlap])
            
            if mismatches <= self.max_mismatches:  # 미스매치 개수로 비교
                if overlap > max_overlap or (overlap == max_overlap and mismatches < min_mismatches):
                    max_overlap = overlap
                    min_mismatches = mismatches
                    best_position = read1_len - overlap
                
        return max_overlap, min_mismatches, best_position

    def merge_sequences(self, seq1: np.ndarray, seq2: np.ndarray, overlap_pos: int) -> np.ndarray:
        """두 시퀀스를 중첩 위치에서 병합하는 함수"""
        overlap_length = len(seq1) - overlap_pos
        # 중첩 부분에서 불일치가 있을 경우 첫 번째 시퀀스의 염기를 우선 사용
        merged = np.concatenate([seq1[:overlap_pos], 
                               seq1[overlap_pos:],
                               seq2[overlap_length:]])
        return merged

    def reconstruct(self, reads_file: str) -> tuple:
        """브루트 포스 방식으로 시퀀스 재구성"""
        print("\n=== 브루트 포스 시퀀스 재구성 시작 ===")
        
        reads_list = []
        for chunk in read_reads_streaming(reads_file, self.read_length):
            reads_list.extend(chunk)
            
        if not reads_list:
            print("리드 데이터를 읽을 수 없습니다.")
            return "", ""
            
        reconstructed = reads_list[0].copy()
        used_reads = {0}
        
        while len(used_reads) < len(reads_list):
            best_overlap = 0
            best_mismatches = float('inf')
            best_read_idx = -1
            best_position = 0
            is_prefix = True
            
            for i in range(len(reads_list)):
                if i in used_reads:
                    continue
                    
                # 접미사 검사
                suffix_overlap, suffix_mismatches, suffix_pos = self.find_best_overlap(reconstructed, reads_list[i])
                if suffix_overlap > best_overlap or (suffix_overlap == best_overlap and suffix_mismatches < best_mismatches):
                    best_overlap = suffix_overlap
                    best_mismatches = suffix_mismatches
                    best_read_idx = i
                    best_position = suffix_pos
                    is_prefix = False
                    
                # 접두사 검사
                prefix_overlap, prefix_mismatches, prefix_pos = self.find_best_overlap(reads_list[i], reconstructed)
                if prefix_overlap > best_overlap or (prefix_overlap == best_overlap and prefix_mismatches < best_mismatches):
                    best_overlap = prefix_overlap
                    best_mismatches = prefix_mismatches
                    best_read_idx = i
                    best_position = prefix_pos
                    is_prefix = True
            
            if best_read_idx == -1 or best_overlap < 10:  # 최소 10bp 이상의 중첩 필요
                break
                
            best_read = reads_list[best_read_idx]
            if is_prefix:
                reconstructed = self.merge_sequences(best_read, reconstructed, best_position)
            else:
                reconstructed = self.merge_sequences(reconstructed, best_read, best_position)
                
            used_reads.add(best_read_idx)
            
            if len(used_reads) % 10 == 0:
                print(f"\r진행률: {(len(used_reads)/len(reads_list))*100:.1f}% ({len(used_reads)}/{len(reads_list)} 리드)", end="")
                
        print(f"\n\n재구성된 시퀀스 길이: {len(reconstructed)}bp")
        
        bin_path, txt_path = save_reconstructed_sequence(reconstructed, reads_file, "brute-force")
        return bin_path, txt_path

if __name__ == "__main__":
    generator = DNASequence()
    bin_path, txt_path = generator.save_sequence(10000, "test.bin")
    reads_bin_path, reads_txt_path = generator.save_reads(bin_path, 50, 25, 10000)
    
    reconstructor = BruteForceReconstructor()
    
    start_time = time.time()
    reconstructed_bin_path, reconstructed_txt_path = reconstructor.reconstruct(reads_bin_path)
    end_time = time.time()
    
    print(f"\n실행 시간: {end_time - start_time:.2f}초")
    
    original_file = bin_path
    reconstructed_file = reconstructed_bin_path
    compare_sequences(original_file, reconstructed_file)

