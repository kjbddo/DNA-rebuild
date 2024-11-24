from typing import List
import numpy as np
from itertools import permutations
from src.reconstructor.dna_reconstruction_interface import DNAReconstructor
from src.sequence_reconstruction import read_reads_streaming

class BruteForceReconstructor(DNAReconstructor):
    def __init__(self, k: int = 31, min_coverage: int = 2, chunk_size: int = 10**6, read_length: int = 100):
        super().__init__(k, min_coverage, chunk_size, read_length)
        self.base_to_num = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        self.num_to_base = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
    
    def reconstruct(self, reads_file: str) -> np.ndarray:
        """Brute Force 방식으로 DNA 시퀀스 재구성"""
        reads = []
        print("리드 데이터 로딩 중...")
        
        # 리드 데이터 로드 및 전처리
        for reads_chunk in read_reads_streaming(reads_file, read_length=self.read_length):
            for read in reads_chunk:
                read_str = ''.join(self.num_to_base[b] for b in read)
                if len(read_str) >= self.k:  # k-mer 길이 이상인 리드만 사용
                    reads.append(read_str)
        
        print(f"총 {len(reads)}개의 리드 로드 완료")
        
        if len(reads) > 10:  # Brute Force는 작은 데이터셋에만 적용
            print("경고: 리드 수가 너무 많습니다. 처음 10개의 리드만 사용합니다.")
            reads = reads[:10]
        
        # 가능한 모든 리드 순서 시도
        print("가능한 모든 조합 시도 중...")
        best_sequence = None
        min_length = float('inf')
        
        for i, perm in enumerate(permutations(reads)):
            if i % 1000 == 0:
                print(f"조합 {i} 시도 중...")
            
            current_sequence = self._merge_reads(list(perm))
            current_length = len(current_sequence)
            
            # 더 짧은 시퀀스 선택 (중복이 더 잘 처리된 것으로 가정)
            if current_length < min_length:
                min_length = current_length
                best_sequence = current_sequence
        
        if not best_sequence:
            print("시퀀스 재구성 실패!")
            return np.array([], dtype=np.uint8)
        
        print("시퀀스 재구성 완료")
        return np.array([self.base_to_num[b] for b in best_sequence], dtype=np.uint8)
    
    def _merge_reads(self, reads: List[str]) -> str:
        """리드들을 하나의 시퀀스로 병합"""
        if not reads:
            return ""
        
        result = reads[0]
        
        for read in reads[1:]:
            max_overlap = 0
            best_score = 0
            
            # 최소 중첩 길이(k) 이상인 경우만 고려
            for i in range(self.k, min(len(result), len(read)) + 1):
                if result.endswith(read[:i]):
                    overlap_score = self._calculate_overlap_score(result[-i:], read[:i])
                    if overlap_score > best_score:
                        max_overlap = i
                        best_score = overlap_score
            
            if max_overlap >= self.k:  # 최소 중첩 길이 조건 확인
                result += read[max_overlap:]
            else:
                result += 'N' + read  # 신뢰할 수 없는 중첩은 구분자로 표시
                
        return result
    
    def _find_overlap(self, str1: str, str2: str) -> int:
        """두 문자열 간의 최대 중첩 길이를 찾음"""
        max_overlap = 0
        for i in range(min(len(str1), len(str2))):
            if str1.endswith(str2[:i+1]):
                max_overlap = i + 1
        return max_overlap
