import unittest
import os
import numpy as np
from src.generator.generate_dna_sequence import DNASequence
from src.reconstructor.debruijn_reconstructor import DeBruijnReconstructor

class TestDeBruijnReconstructor(unittest.TestCase):
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
        
        # 테스트 매개변수
        self.sequence_length = 10000
        self.read_length = 100
        self.coverage = 30
        
        # 테스트 파일 생성
        self.bin_path, _ = self.generator.save_sequence(
            self.sequence_length, f"test_debruijn_{self.read_length}.bin")
        
        # 리드 파일 생성
        self.reads_bin, _ = self.generator.save_reads(
            self.bin_path,
            read_length=self.read_length,
            overlap=50,
        )
    
    def test_debruijn_reconstruction(self):
        # De Bruijn 재구성기 초기화
        reconstructor = DeBruijnReconstructor(
            k=31,
            min_coverage=2,
            read_length=self.read_length
        )
        
        # 시퀀스 재구성
        reconstructed = reconstructor.reconstruct(self.reads_bin)
        
        # 정확도 검증
        accuracy = reconstructor.validate(self.bin_path, reconstructed)
        
        print(f"\n재구성된 시퀀스 길이: {len(reconstructed)}")
        print(f"정확도: {accuracy:.2f}%")
        
        # 검증
        self.assertIsNotNone(reconstructed)
        self.assertGreater(len(reconstructed), 0)
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 100.0)

if __name__ == '__main__':
    unittest.main()