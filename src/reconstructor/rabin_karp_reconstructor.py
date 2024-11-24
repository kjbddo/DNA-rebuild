from typing import List
import numpy as np
from src.reconstructor.dna_reconstruction_interface import DNAReconstructor
from src.sequence_reconstruction import read_reads_streaming

class RabinKarpReconstructor(DNAReconstructor):
    def __init__(self, k: int = 31, min_coverage: int = 2, chunk_size: int = 10**6, read_length: int = 100):
        super().__init__(k, min_coverage, chunk_size, read_length)
        self.base_to_num = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        self.num_to_base = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
        self.prime = 101
        self.base = 256
    
    def reconstruct(self, reads_file: str) -> np.ndarray:
        """Rabin-Karp 알고리즘을 사용한 시퀀스 재구성"""
        reads = []
        print("리드 데이터 로딩 중...")
        
        for reads_chunk in read_reads_streaming(reads_file, read_length=self.read_length):
            for read in reads_chunk:
                read_str = ''.join(self.num_to_base[b] for b in read)
                reads.append(read_str)
        
        print(f"총 {len(reads)}개의 리드 로드 완료")
        
        # 첫 번째 리드로 시작
        reconstructed = reads[0]
        
        print("시퀀스 재구성 중...")
        for i, read in enumerate(reads[1:], 1):
            if i % 1000 == 0:
                print(f"진행률: {i/len(reads)*100:.1f}%")
            
            overlap = self._find_overlap(reconstructed, read)
            reconstructed += read[overlap:]
        
        print("시퀀스 재구성 완료")
        return np.array([self.base_to_num[b] for b in reconstructed], dtype=np.uint8)
    
    def _calculate_hash(self, s: str) -> int:
        """문자열의 해시값 계산"""
        h = 0
        for c in s:
            h = (h * self.base + ord(c)) % self.prime
        return h
    
    def _find_overlap(self, text: str, pattern: str) -> int:
        """Rabin-Karp를 사용하여 두 문자열 간의 최대 중첩 찾기"""
        min_length = min(len(text), len(pattern))
        if min_length == 0:
            return 0
        
        max_overlap = 0
        
        # 패턴의 접두사들의 해시값 계산
        pattern_hash = 0
        text_hash = 0
        h = 1
        
        for i in range(min_length):
            pattern_hash = (pattern_hash * self.base + ord(pattern[i])) % self.prime
            text_hash = (text_hash * self.base + ord(text[-(min_length-i)])) % self.prime
            if i > 0:
                h = (h * self.base) % self.prime
        
        for i in range(min_length):
            if pattern_hash == text_hash:
                return min_length - i
            
            if i < len(text) - min_length:
                text_hash = (text_hash - ord(text[i]) * h) * self.base + ord(text[i + min_length])
                text_hash %= self.prime
        
        return 0
