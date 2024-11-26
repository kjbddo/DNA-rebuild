import os
from datetime import datetime
from src.generator.generate_dna_sequence import DNASequence
from src.reconstructor.debruijn_reconstructor import DeBruijnReconstructor
from src.reconstructor.rabin_karp_reconstructor import RabinKarpReconstructor
from src.reconstructor.kmp_reconstructor import KMPReconstructor
from src.reconstructor.brute_force_reconstructor import BruteForceReconstructor

def reconstruct_dna(method: str, original_file: str, reads_file: str, k: int = 31, min_coverage: int = 2, read_length: int = 100):
    reconstructors = {
        'debruijn': DeBruijnReconstructor(k, min_coverage, read_length=read_length),
        'rabin-karp': RabinKarpReconstructor(k, min_coverage, read_length=read_length),
        'kmp': KMPReconstructor(k, min_coverage, read_length=read_length),
        'brute-force': BruteForceReconstructor(k, min_coverage, read_length=read_length)
    }

    reconstructor = reconstructors[method]
    reconstructed = reconstructor.reconstruct(reads_file)
    accuracy = reconstructor.validate(original_file, reconstructed)
    
    return reconstructed, accuracy 

if __name__ == "__main__":
    # 프로젝트 루트 디렉토리 설정
    project_root = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(project_root, 'data')
    
    # DNA 시퀀스 생성
    generator = DNASequence(seed=42)
    sequence_length = 10000  # 10Kbp
    bin_path, txt_path = generator.save_sequence(sequence_length, "test_sequence.bin")
    
    # 리드 생성
    reads_path = generator.save_reads(
        sequence_file=bin_path,
        read_length=100,    # 100bp 리드
        overlap=50         # 50bp overlap
    )
    
    # 각 알고리즘으로 재구성 테스트
    methods = ['debruijn', 'rabin-karp', 'kmp', 'brute-force']
    
    for method in methods:
        print(f"\n=== {method} 알고리즘 테스트 ===")
        reconstructed, accuracy = reconstruct_dna(
            method=method,
            original_file=bin_path,
            reads_file=reads_path,
            k=31,
            min_coverage=2,
            read_length=100
        )
        print(f"정확도: {accuracy:.2f}%")
