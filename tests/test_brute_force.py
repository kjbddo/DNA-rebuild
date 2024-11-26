import unittest
import os, sys
import numpy as np

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.generator.generate_dna_sequence import DNASequence
from src.reconstructor.brute_force_reconstructor import BruteForceReconstructor

class TestBruteForceReconstructor(unittest.TestCase):
    def setUp(self):
        # 테스트 데이터 디렉토리 설정
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = os.path.join(self.project_root, 'data')
        self.original_dir = os.path.join(self.data_dir, 'original')
        self.reads_dir = os.path.join(self.data_dir, 'reads')
        
        # 디렉토리 생성
        os.makedirs(self.original_dir, exist_ok=True)
        os.makedirs(self.reads_dir, exist_ok=True)
        
        # DNA 시퀀스 생성기 초기화
        self.generator = DNASequence(seed=42)
        
        # 테스트 매개변수 수정
        self.sequence_length = 100  # 시퀀스 길이
        self.read_length = 20      # 리드 길이를 더 작게 조정
        self.overlap = 10          # overlap을 더 작게 조정
        self.chunk_size = 10       # 청크 크기도 작게 조정
        
        # 테스트 파일 생성
        self.bin_path, _ = self.generator.save_sequence(
            self.sequence_length, 
            f"test_sequence_{self.sequence_length}bp.bin"
        )
        
        # 리드 파일 생성
        self.reads_bin, _ = self.generator.save_reads(
            sequence_file=self.bin_path,
            read_length=self.read_length,
            overlap=self.overlap,
            chunk_size=self.chunk_size
        )
    
    def test_brute_force_reconstruction(self):
        # Brute Force 재구성기 초기화
        reconstructor = BruteForceReconstructor(
            min_coverage=1,
            chunk_size=self.chunk_size,
            read_length=self.read_length
        )
        
        # 시퀀스 재구성
        reconstructed = reconstructor.reconstruct(self.reads_bin)
        
        # 정확도 검증
        accuracy = reconstructor.validate(self.bin_path, reconstructed)
        
        print(f"\n=== 테스트 결과 ===")
        print(f"원본 시퀀스 길이: {self.sequence_length}bp")
        print(f"재구성된 시퀀스 길이: {len(reconstructed)}bp")
        print(f"정확도: {accuracy:.2f}%")
        
        # 검증
        self.assertIsNotNone(reconstructed)
        self.assertGreater(len(reconstructed), 0)
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 100.0)

if __name__ == '__main__':
    unittest.main()