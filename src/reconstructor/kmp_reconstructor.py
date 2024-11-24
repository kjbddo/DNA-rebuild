from typing import List
import numpy as np
from src.reconstructor.dna_reconstruction_interface import DNAReconstructor
from src.sequence_reconstruction import read_reads_streaming

class KMPReconstructor(DNAReconstructor):
    def __init__(self, k: int = 31, min_coverage: int = 2, chunk_size: int = 10**6, read_length: int = 100):
        super().__init__(k, min_coverage, chunk_size, read_length)
        self.base_to_num = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        self.num_to_base = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
    
    def reconstruct(self, reads_file: str) -> np.ndarray:
        """KMP 알고리즘을 사용한 시퀀스 재구성"""
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
            
            overlap = self._find_max_overlap(reconstructed, read)
            reconstructed += read[overlap:]
        
        print("시퀀스 재구성 완료")
        return np.array([self.base_to_num[b] for b in reconstructed], dtype=np.uint8)
    
    def _compute_lps(self, pattern: str) -> List[int]:
        """LPS(Longest Proper Prefix which is also Suffix) 배열 계산"""
        length = len(pattern)
        lps = [0] * length
        
        len_p = 0
        i = 1
        
        while i < length:
            if pattern[i] == pattern[len_p]:
                len_p += 1
                lps[i] = len_p
                i += 1
            else:
                if len_p != 0:
                    len_p = lps[len_p - 1]
                else:
                    lps[i] = 0
                    i += 1
        
        return lps
    
    def _find_max_overlap(self, text: str, pattern: str) -> int:
        """두 문자열 간의 최대 중첩 길이를 찾음"""
        min_length = min(len(text), len(pattern))
        pattern_prefix = pattern[:min_length]
        text_suffix = text[-min_length:]
        
        lps = self._compute_lps(pattern_prefix)
        
        i = 0  # text_suffix 인덱스
        j = 0  # pattern_prefix 인덱스
        max_overlap = 0
        
        while i < len(text_suffix):
            if text_suffix[i] == pattern_prefix[j]:
                i += 1
                j += 1
                max_overlap = max(max_overlap, j)
            else:
                if j != 0:
                    j = lps[j - 1]
                else:
                    i += 1
        
        return max_overlap
