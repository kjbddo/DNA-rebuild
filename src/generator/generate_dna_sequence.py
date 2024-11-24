import numpy as np
import random
import os
from typing import Generator, BinaryIO, Tuple, List

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
        """시퀀스를 청크 단위로 생성하여 파일에 저장"""
        # 데이터 디렉토리 경로 설정
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        data_dir = os.path.join(project_root, 'data')
        original_dir = os.path.join(data_dir, 'original')
        
        # 디렉토리가 없으면 생성
        os.makedirs(original_dir, exist_ok=True)
        
        # 타임스탬프를 포함한 파일명 생성
  
        base_filename = f"sequence_{total_length}bp"
        bin_path = os.path.join(original_dir, f"{base_filename}.bin")
        txt_path = os.path.join(original_dir, f"{base_filename}_first10000bp.txt")
        
        # 동일한 길이의 가장 최근 파일 찾기
        existing_files = [f for f in os.listdir(original_dir) if f.startswith(f"sequence_{total_length}bp_") and f.endswith('.bin')]
        if existing_files:
            latest_file = max(existing_files)
            existing_bin = os.path.join(original_dir, latest_file)
            existing_txt = os.path.join(original_dir, latest_file.replace('.bin', '_first10000bp.txt'))
            
            if os.path.exists(existing_bin) and os.path.exists(existing_txt):
                print(f"\n동일한 길이의 기존 시퀀스 파일을 발견했습니다:")
                print(f"바이너리 파일: {existing_bin}")
                print(f"텍스트 파일: {existing_txt}")
                return existing_bin, existing_txt
        
        with open(bin_path, 'wb') as f:
            np.array([total_length], dtype=np.uint64).tofile(f)
            
            for i, chunk in enumerate(self.generate_sequence_chunks(total_length, chunk_size)):
                if i == 0:
                    first_100bp = chunk[:100].copy()
                    first_10000bp = chunk[:min(10000, len(chunk))].copy()
                    print(f"\n원본 시퀀스 (첫 100bp): {self.sequence_to_string(first_100bp)}")
                
                if i == (total_length - 1) // chunk_size:
                    last_100bp = chunk[-100:].copy()
                    print(f"원본 시퀀스 (마지막 100bp): {self.sequence_to_string(last_100bp)}")
                
                chunk.tofile(f)
        
        # 첫 10000bp를 텍스트 파일로 저장
        with open(txt_path, 'w') as f:
            f.write(self.sequence_to_string(first_10000bp))
        
        print(f"\n시퀀스가 {bin_path}에 저장되었습니다.")
        print(f"첫 10000bp가 {txt_path}에 저장되었습니다.")
        return bin_path, txt_path

    def generate_reads_stream(self, file: BinaryIO, total_length: int, read_length: int,
                            coverage: int = 30) -> Generator[np.ndarray, None, None]:
        """스트리밍 방식으로 리드 생성"""
        read_count = (total_length * coverage) // read_length
        max_start_pos = total_length - read_length
        step_size = max(1, max_start_pos // (read_count - 1))
        
        # 첫 번째와 마지막 리드 표시를 위한 변수
        first_read = True
        
        for start_pos in range(0, max_start_pos + 1, step_size):
            file.seek(8 + start_pos)  # 8바이트 헤더 건너뛰기
            read = np.fromfile(file, dtype=np.uint8, count=read_length)
            
            if len(read) == read_length:
                if first_read:
                    print(f"첫 번째 리드: {self.sequence_to_string(read)}")
                    first_read = False
                yield read

    def save_reads(self, sequence_file: str, read_length: int = 100, coverage: int = 30, chunk_size: int = 1000) -> Tuple[str, str]:
        """시퀀스로부터 리드를 생성하고 저장 (청크 단위로 처리)"""
        # 프로젝트 루트 디렉토리 설정
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        reads_dir = os.path.join(project_root, 'data', 'reads')
        os.makedirs(reads_dir, exist_ok=True)
        
        # 리드 길이를 포함한 파일명 생성
        base_filename = f"reads_{read_length}bp"
        reads_bin = os.path.join(reads_dir, f"{base_filename}.bin")
        reads_txt = os.path.join(reads_dir, f"{base_filename}.txt")
        
        # 동일한 리드 길이의 가장 최근 파일 찾기
        existing_files = [f for f in os.listdir(reads_dir) if f.startswith(f"reads_{read_length}bp_") and f.endswith('.bin')]
        if existing_files:
            latest_file = max(existing_files)
            existing_bin = os.path.join(reads_dir, latest_file)
            existing_txt = os.path.join(reads_dir, latest_file.replace('.bin', '.txt'))
            
            if os.path.exists(existing_bin) and os.path.exists(existing_txt):
                print(f"\n동일한 리드 길이의 기존 파일을 발견했습니다:")
                print(f"바이너리 파일: {existing_bin}")
                print(f"텍스트 파일: {existing_txt}")
                return existing_bin, existing_txt
        
        try:
            with open(sequence_file, 'rb') as f:
                # 시퀀스 길이 읽기
                total_length = np.fromfile(f, dtype=np.uint64, count=1)[0]
                read_count = (total_length * coverage) // read_length
                
                print(f"\n리드 생성 시작:")
                print(f"총 시퀀스 길이: {total_length:,}bp")
                print(f"리드 길이: {read_length}bp")
                print(f"목표 커버리지: {coverage}x")
                print(f"생성할 총 리드 수: {read_count:,}")
                
                # 헤더 정보 저장
                with open(reads_bin, 'wb') as bin_f:
                    np.array([read_length, read_count], dtype=np.uint64).tofile(bin_f)
                
                # 청크 단위로 리드 생성 및 저장
                current_read = 0
                chunk_reads = []
                last_progress = 0
                
                for read in self.generate_reads_stream(f, total_length, read_length, coverage):
                    chunk_reads.append(read)
                    current_read += 1
                    
                    # 진행률 표시 (1% 단위로)
                    progress = (current_read * 100) // read_count
                    if progress > last_progress:
                        print(f"진행률: {progress}% ({current_read:,}/{read_count:,} 리드 생성됨)", end='\r')
                        last_progress = progress
                    
                    # 청크가 차면 파일에 저장
                    if len(chunk_reads) >= chunk_size:
                        self._save_reads_chunk(chunk_reads, reads_bin, reads_txt)
                        chunk_reads = []
                
                # 남은 리드 저장
                if chunk_reads:
                    self._save_reads_chunk(chunk_reads, reads_bin, reads_txt)
                
                print(f"\n\n리드 생성 완료!")
                print(f"바이너리 파일: {reads_bin}")
                print(f"텍스트 파일: {reads_txt}")
                
                return reads_bin, reads_txt
                
        except FileNotFoundError:
            print(f"시퀀스 파일을 찾을 수 없습니다: {sequence_file}")
            return "", ""
        except Exception as e:
            print(f"리드 생성 중 오류 발생: {str(e)}")
            return "", ""

    def _save_reads_chunk(self, chunk_reads: List[np.ndarray], bin_path: str, txt_path: str):
        """청크 단위로 리드를 파일에 저장"""
        # 바이너리로 저장
        with open(bin_path, 'ab') as bin_f:
            for read in chunk_reads:
                read.tofile(bin_f)
        
        # 텍스트로 저장
        with open(txt_path, 'a') as txt_f:
            for read in chunk_reads:
                read_str = ''.join(self.base_map[b] for b in read)
                txt_f.write(read_str + '\n')

if __name__ == "__main__":
    generator = DNASequence()
    bin_path, txt_path = generator.save_sequence(10000, "test.bin")
    print(f"생성된 파일: {bin_path}, {txt_path}")
    