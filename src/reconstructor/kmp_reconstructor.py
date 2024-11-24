from typing import List
import numpy as np
from src.reconstructor.dna_reconstruction_interface import DNAReconstructor
from src.sequence_reconstruction import read_reads_streaming

class KMPReconstructor(DNAReconstructor):
    def __init__(self, k: int = 31, min_coverage: int = 2, chunk_size: int = 10**6, read_length: int = 100):
        super().__init__(k, min_coverage, chunk_size, read_length)
        self.base_to_num = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
        self.num_to_base = {0: 'A', 1: 'T', 2: 'C', 3: 'G'}
    
    def reconstruct(self, reads_file: str) -> np.ndarray:
        """KMP 알고리즘을 사용하여 DNA 시퀀스 재구성"""
        reads = []
        print("리드 데이터 로딩 중...")
        
        # 바이너리 파일에서 리드 데이터 로드
        for reads_chunk in read_reads_streaming(reads_file, read_length=self.read_length):
            reads.extend(reads_chunk)
        
        print(f"총 {len(reads)}개의 리드 로드 완료")
        
        # 리드를 문자열로 변환
        reads_str = [self._convert_read_to_string(read) for read in reads]
        
        # KMP 알고리즘으로 시퀀스 재구성
        sequence = self._reconstruct_sequence(reads_str)
        
        if not sequence:
            print("시퀀스 재구성 실패!")
            return np.array([], dtype=np.uint8)
        
        print("시퀀스 재구성 완료")
        return np.array([self.base_to_num[b] for b in sequence], dtype=np.uint8)
    
    def _convert_read_to_string(self, read: np.ndarray) -> str:
        """숫자 배열을 염기서열 문자열로 변환"""
        return ''.join(self.num_to_base[b] for b in read)
    
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
