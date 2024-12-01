import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.sequence_comparison import compare_sequences
from src.generator.generate_dna_sequence import DNASequence
from typing import List, Dict, Optional, Tuple
import numpy as np
from src.sequence_reconstruction import (
    read_reads_streaming,
    save_reconstructed_sequence
)

class RabinKarpReconstructor:
    def __init__(self, read_length, max_mismatches, chunk_size):
        self.read_length = read_length
        self.max_mismatches = max_mismatches
        self.chunk_size = chunk_size
        self.base_map = {0: 'A', 1: 'T', 2: 'C', 3: 'G'}
        self.base_reverse_map = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
        self.prime = 101
        self.base = 4     # DNA는 A,T,C,G 4개 문자만 사용

    def calculate_hash(self, sequence: np.ndarray, length: int) -> int:
        """주어진 시퀀스의 해시값 계산"""
        hash_value = 0
        for i in range(length):
            hash_value = (hash_value * self.base + int(sequence[i])) % self.prime
        return hash_value

    def recalculate_hash(self, old_hash: int, old_char: int, new_char: int, pattern_length: int) -> int:
        """이전 해시값을 이용하여 새로운 해시값 계산"""
        # 이전 첫 문자의 값을 제거
        hash_value = (old_hash - int(old_char) * pow(self.base, pattern_length-1, self.prime)) % self.prime
        # 새로운 문자 추가
        hash_value = (hash_value * self.base + int(new_char)) % self.prime
        return hash_value if hash_value >= 0 else hash_value + self.prime

    def find_overlap_rabin_karp(self, read1: np.ndarray, read2: np.ndarray, min_overlap: int = 10) -> int:
        """Rabin-Karp 알고리즘을 사용하여 두 리드 간의 최대 중첩 길이를 찾음 (미스매치 허용)"""
        read1_len = len(read1)
        read2_len = len(read2)
        max_overlap = 0
        
        for overlap_len in range(min(read1_len, read2_len), min_overlap-1, -1):
            pattern = read2[:overlap_len]
            text = read1[-overlap_len:]
            
            pattern_hash = self.calculate_hash(pattern, overlap_len)
            text_hash = self.calculate_hash(text, overlap_len)
            
            # 해시값이 같거나 시퀀스가 허용된 미스매치 수 이내인 경우
            if pattern_hash == text_hash or self._check_mismatches(pattern, text):
                max_overlap = overlap_len
                break
                
        return max_overlap

    def _check_mismatches(self, seq1: np.ndarray, seq2: np.ndarray) -> bool:
        """두 시퀀스 간의 미스매치 수를 확인"""
        mismatches = np.sum(seq1 != seq2)
        return mismatches <= self.max_mismatches

    def reconstruct(self, reads_file: str) -> np.ndarray:
        """Rabin-Karp 알고리즘을 사용한 시퀀스 재구성"""
        print("\n=== Rabin-Karp 시퀀스 재구성 시작 ===")
        
        # 리드 데이터 읽기
        reads_list = []
        for chunk in read_reads_streaming(reads_file, self.read_length, self.chunk_size):
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
                suffix_overlap = self.find_overlap_rabin_karp(reconstructed, reads_list[i])
                if suffix_overlap > best_overlap:
                    best_overlap = suffix_overlap
                    best_read_idx = i
                    is_prefix = False
                    
                # 접두사 중첩 확인
                prefix_overlap = self.find_overlap_rabin_karp(reads_list[i], reconstructed)
                if prefix_overlap > best_overlap:
                    best_overlap = prefix_overlap
                    best_read_idx = i
                    is_prefix = True
            
            if best_read_idx == -1:
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
        bin_path, txt_path = save_reconstructed_sequence(reconstructed, reads_file, "rabin-karp")
        
        return bin_path, txt_path

if __name__ == "__main__":
    read_length = 50
    max_mismatches = 2
    chunk_size = 10000
    overlap = 25
    
    generator = DNASequence()
    bin_path, txt_path = generator.save_sequence(chunk_size, "test.bin")
    reads_bin_path, reads_txt_path = generator.save_reads(bin_path, read_length, overlap, chunk_size)
    reconstructor = RabinKarpReconstructor(read_length, max_mismatches, chunk_size)
    
    start_time = time.time()
    reconstructed_bin_path, reconstructed_txt_path = reconstructor.reconstruct(reads_bin_path)
    end_time = time.time()
    
    print(f"\n실행 시간: {end_time - start_time:.2f}초")
    
    original_file = bin_path
    reconstructed_file = reconstructed_bin_path
    compare_sequences(original_file, reconstructed_file)
