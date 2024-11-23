import numpy as np
from typing import BinaryIO, Generator, List, Dict, Tuple
import random

class DNASequence:
    def __init__(self, seed: int = None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        self.base_map = {0: 'A', 1: 'T', 2: 'C', 3: 'G'}
        self.base_reverse_map = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
    
    def sequence_to_string(self, sequence: np.ndarray) -> str:
        """숫자 시퀀스를 염기서열 문자열로 변환"""
        if sequence is None:
            return ""
        return ''.join(self.base_map[b] for b in sequence)
    
    def generate_sequence_chunks(self, total_length: int, chunk_size: int = 10**6) -> Generator[np.ndarray, None, None]:
        """청크 단위로 무작위 DNA 시퀀스 생성"""
        rng = np.random.Generator(np.random.PCG64())
        for i in range(0, total_length, chunk_size):
            current_chunk_size = min(chunk_size, total_length - i)
            chunk = rng.integers(0, 4, size=current_chunk_size, dtype=np.uint8)
            yield chunk
    
    def save_sequence(self, total_length: int, filename: str, chunk_size: int = 10**6):
        """시퀀스를 청크 단위로 생성하여 파일에 저장하고, 첫 10000bp를 텍스트 파일로도 저장"""
        first_10000bp = None
        first_100bp = None
        last_100bp = None
        
        with open(filename, 'wb') as f:
            np.array([total_length], dtype=np.uint64).tofile(f)
            
            for i, chunk in enumerate(self.generate_sequence_chunks(total_length, chunk_size)):
                if i == 0:
                    first_20bp = chunk[:100].copy()
                    # 첫 10000bp 저장 (또는 전체 길이가 10000bp 미만인 경우 전체)
                    first_10000bp = chunk[:min(10000, len(chunk))].copy()
                    print(f"\n원본 시퀀스 (첫 100bp): {self.sequence_to_string(first_20bp)}")
                
                if i == (total_length - 1) // chunk_size:
                    last_20bp = chunk[-100:].copy()
                    print(f"원본 시퀀스 (마지막 100bp): {self.sequence_to_string(last_20bp)}")
                
                chunk.tofile(f)
        
        # 첫 10000bp를 텍스트 파일로 저장
        txt_filename = filename.rsplit('.', 1)[0] + '_first10000bp.txt'
        with open(txt_filename, 'w') as f:
            f.write(self.sequence_to_string(first_10000bp))
        
        print(f"\n첫 10000bp가 {txt_filename}에 저장되었습니다.")
        return first_20bp, last_20bp
    
    def generate_reads_stream(self, file: BinaryIO, total_length: int, read_length: int,
                            coverage: int = 30) -> Generator[np.ndarray, None, None]:
        """스트리밍 방식으로 리드 생성"""
        read_count = (total_length * coverage) // read_length
        max_start_pos = total_length - read_length
        step_size = max(1, max_start_pos // (read_count - 1))
        
        first_read = None
        last_read = None
        current_read_count = 0
        
        # 균일 간격으로 리드 생성
        for start_pos in range(0, max_start_pos + 1, step_size):
            file.seek(8 + start_pos)
            read = np.fromfile(file, dtype=np.uint8, count=read_length)
            
            if len(read) == read_length:
                if current_read_count == 0:
                    first_read = read.copy()
                    print(f"\n첫 번째 리드: {self.sequence_to_string(first_read)}")
                
                current_read_count += 1
                last_read = read.copy()
                yield read
        
        print(f"마지막 리드: {self.sequence_to_string(last_read)}")

    def read_reads_streaming(filename: str, chunk_size: int = 1000) -> Generator[List[np.ndarray], None, Tuple[int, int]]:
        """리드를 청크 단위로 읽어오는 제너레이터"""
        with open(filename, 'rb') as f:
            # 메타데이터 읽기
            read_count, read_length = np.fromfile(f, dtype=np.uint32, count=2)
            
            reads = []
            for _ in range(read_count):
                read = np.fromfile(f, dtype=np.uint8, count=read_length)
                if len(read) == read_length:
                    reads.append(read)
                    
                    if len(reads) >= chunk_size:
                        yield reads
                        reads = []
            
            if reads:  # 남은 리드들 처리
                yield reads
                
        return read_count, read_length


if __name__ == "__main__":
    generator = DNASequence()
    generator.save_sequence(100000, "test.bin")
    