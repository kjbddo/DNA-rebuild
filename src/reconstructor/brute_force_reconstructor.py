from typing import List, Tuple
import numpy as np
from itertools import permutations
from src.reconstructor.dna_reconstruction_interface import DNAReconstructor
from src.sequence_reconstruction import read_reads_streaming, save_reconstructed_sequence
import os

class BruteForceReconstructor(DNAReconstructor):
    def __init__(self, min_coverage: int = 2, chunk_size: int = 10**6, read_length: int = 100):
        super().__init__(min_coverage, chunk_size, read_length)
        self.base_to_num = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
        self.num_to_base = {0: 'A', 1: 'T', 2: 'C', 3: 'G'}
    
    def reconstruct(self, reads_file: str) -> np.ndarray:
        """Brute Force 방식으로 DNA 시퀀스 재구성"""
        reads = []
        print("\n=== 리드 데이터 로딩 ===")
        
        # 리드 데이터 로드 및 전처리
        for reads_chunk in read_reads_streaming(reads_file, read_length=self.read_length):
            # 디버깅을 위한 출력
            if len(reads) == 0:
                print(f"첫 번째 청크 크기: {len(reads_chunk)}")
                if len(reads_chunk) > 0:
                    first_read = reads_chunk[0]
                    print(f"첫 번째 리드 길이: {len(first_read)}bp")
                    print(f"첫 번째 리드 데이터: {first_read}")
                    print(f"첫 번째 리드 문자열: {self._convert_read_to_string(first_read)}")
            reads.extend(reads_chunk)
        
        print(f"총 {len(reads)}개의 리드 로드 완료")
        
        # 리드를 문자열로 변환
        reads_str = [self._convert_read_to_string(read) for read in reads]
        
        # 처음 몇 개의 리드 출력하여 확인
        print("\n처음 5개 리드:")
        for i, read in enumerate(reads_str[:5]):
            print(f"리드 {i}: {read}")
        
        # 첫 번째 리드로 시작
        result_sequence = reads_str[0]
        used_reads = {0}  # 사용된 리드의 인덱스를 추적
        
        print("\n=== 시퀀스 재구성 시작 ===")
        iteration = 0
        
        while len(used_reads) < len(reads_str):
            iteration += 1
            print(f"\n반복 {iteration}:")
            
            best_overlap = 0
            best_score = 0
            best_read_idx = -1
            best_is_prefix = True
            
            # 현재 시퀀스의 상태 출력
            print(f"현재 시퀀스 길이: {len(result_sequence)}bp")
            print(f"현재 시퀀스 시작: {result_sequence[:50]}...")
            print(f"현재 시퀀스 끝: ...{result_sequence[-50:]}")
            
            # 아직 사용하지 않은 모든 리드에 대해
            for i, read in enumerate(reads_str):
                if i in used_reads:
                    continue
                
                # 현재 시퀀스의 접미사와 새 리드의 접두사 비교
                suffix_overlap = self._find_best_overlap(result_sequence, read)
                if suffix_overlap[0] > best_overlap and suffix_overlap[1] >= 0.9:  # 90% 이상 일치하는 경우만
                    best_overlap = suffix_overlap[0]
                    best_score = suffix_overlap[1]
                    best_read_idx = i
                    best_is_prefix = True
                    print(f"  발견된 접미사 중첩 (리드 {i}):")
                    print(f"  현재 시퀀스 끝: ...{result_sequence[-best_overlap:]}")
                    print(f"  다음 리드 시작: {read[:best_overlap]}...")
                
                # 현재 시퀀스의 접두사와 새 리드의 접미사 비교
                prefix_overlap = self._find_best_overlap(read, result_sequence)
                if prefix_overlap[0] > best_overlap and prefix_overlap[1] >= 0.9:
                    best_overlap = prefix_overlap[0]
                    best_score = prefix_overlap[1]
                    best_read_idx = i
                    best_is_prefix = False
                    print(f"  발견된 접두사 중첩 (리드 {i}):")
                    print(f"  다음 리드 끝: ...{read[-best_overlap:]}")
                    print(f"  현재 시퀀스 시작: {result_sequence[:best_overlap]}...")
            
            if best_read_idx == -1:
                print("\n더 이상 적절한 중첩을 찾을 수 없음")
                break
            
            # 선택된 리드를 시퀀스에 추가
            next_read = reads_str[best_read_idx]
            old_length = len(result_sequence)
            
            if best_is_prefix:
                result_sequence += next_read[best_overlap:]
            else:
                result_sequence = next_read[:-best_overlap] + result_sequence
            
            used_reads.add(best_read_idx)
            print(f"\n리드 {best_read_idx} 추가됨:")
            print(f"이전 길이: {old_length}bp -> 새 길이: {len(result_sequence)}bp")
            print(f"사용된 리드: {len(used_reads)}/{len(reads_str)}")
        
        print("\n=== 재구성 완료 ===")
        print(f"최종 시퀀스 길이: {len(result_sequence)}bp")
        print(f"사용된 리드 수: {len(used_reads)}/{len(reads_str)}")
        
        # 재구성된 시퀀스를 numpy 배열로 변환
        reconstructed = np.array([self.base_to_num[b] for b in result_sequence], dtype=np.uint8)
        
        # 재구성된 시퀀스 저장
        save_reconstructed_sequence(reconstructed, reads_file, method="brute-force")
        
        return reconstructed
        
    def _convert_read_to_string(self, read: np.ndarray) -> str:
        """숫자 배열을 염기서열 문자열로 변환"""
        return ''.join(self.num_to_base[b] for b in read)
    
    def _find_best_overlap(self, seq1: str, seq2: str) -> Tuple[int, float]:
        """두 시퀀스 간의 최적의 중첩 길이와 점수를 찾음"""
        max_overlap = 0
        best_score = 0
        
        # 최소 중첩 길이 설정 (너무 짧은 중첩은 피함)
        min_overlap = self.read_length // 3
        
        # 가능한 모든 중첩 길이에 대해 검사
        for overlap_len in range(min_overlap, min(len(seq1), len(seq2)) + 1):
            # seq1의 끝부분과 seq2의 시작부분 비교
            suffix = seq1[-overlap_len:]
            prefix = seq2[:overlap_len]
            
            if suffix == prefix:  # 정확히 일치하는 경우
                score = 1.0
            else:  # 부분적으로 일치하는 경우
                matches = sum(1 for x, y in zip(suffix, prefix) if x == y)
                score = matches / overlap_len
            
            if score > best_score:
                max_overlap = overlap_len
                best_score = score
        
        return max_overlap, best_score
    
    def _calculate_overlap_score(self, seq1: str, seq2: str) -> float:
        """두 시퀀스 간의 겹침 점수를 계산"""
        if len(seq1) != len(seq2):
            return 0
        
        # 일치하는 염기 수를 세고 정규화된 점수 반환
        matches = sum(1 for a, b in zip(seq1, seq2) if a == b)
        return matches / len(seq1)  # 0.0 ~ 1.0 사이의 점수
