import glob
import numpy as np
import random
import os
from typing import Generator, BinaryIO, TextIO, Tuple, List

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
        
        # 타일명 생성
        base_filename = f"sequence_{total_length}bp"
        bin_path = os.path.join(original_dir, f"{base_filename}.bin")
        txt_path = os.path.join(original_dir, f"{base_filename}_first10000bp.txt")
        
        # 기존 파일 확인
        if os.path.exists(bin_path) and os.path.exists(txt_path):
            print(f"\n동일한 길이의 기존 시퀀스 파일이 존재합니다:")
            print(f"바이너리 파일: {bin_path}")
            print(f"텍스트 파일: {txt_path}")
            return bin_path, txt_path
        
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
                             overlap: int = 50) -> Generator[np.ndarray, None, None]:
        """스트리밍 방식으로 리드 생성 (overlap 기반)
        
        Args:
            file: 바이너리 파일 객체
            total_length: 전체 시퀀스 길이
            read_length: 각 리드의 길이
            overlap: 연속된 리드 간의 겹치는 염기 수
        """
        try:
            # overlap 유효성 검사
            if overlap >= read_length:
                raise ValueError(f"overlap({overlap})이 read_length({read_length})보다 크거나 같습니다")
            
            # 각 리드의 시작 위치 계산
            step_size = read_length - overlap
            # range에 들어가는 값들을 정수형으로 명시적 변환
            start = 0
            # 마지막 리드가 시퀀스의 끝부분을 포함하도록 end 위치 수정
            end = total_length - read_length + 1
            step = int(step_size)
            read_positions = list(range(start, end, step))
            
            # 마지막 위치가 시퀀스의 끝부분을 포함하지 않는 경우, 마지막 리드 추가
            if read_positions[-1] + read_length < total_length:
                read_positions.append(total_length - read_length)
            
            first_read = True
            for start_pos in read_positions:
                # 헤더(8바이트) 이후부터 읽기
                file.seek(8 + start_pos)
                read = np.fromfile(file, dtype=np.uint8, count=read_length)
                
                if len(read) == read_length:
                    if first_read:
                        print(f"\n첫 번째 리드: {self.sequence_to_string(read)}")
                        first_read = False
                    yield read
                else:
                    print(f"\n경고: 불완전한 리드 발견 (위치: {start_pos}, 길이: {len(read)})")
                    
        except Exception as e:
            print(f"리드 생성 중 오류 발생: {str(e)}")

    def save_reads(self, sequence_file: str, read_length: int = 100, overlap: int = 50, chunk_size: int = 1000) -> Tuple[str, str]:
        """시퀀스로부터 리드를 생성하고 저장 (overlap 기반)"""
        try:
            with open(sequence_file, 'rb') as f:
                total_length = int(np.fromfile(f, dtype=np.uint64, count=1)[0])
                
                # 데이터 디렉토리 설정
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                reads_dir = os.path.join(project_root, 'data', 'reads')
                os.makedirs(reads_dir, exist_ok=True)
                
                # 파일명 생성 (리드 길이와 overlap 정보 포함)
                base_filename = f"sequence_{total_length}bp_reads_{read_length}bp_overlap_{overlap}bp"
                reads_bin = os.path.join(reads_dir, f"{base_filename}.bin")
                reads_txt = os.path.join(reads_dir, f"{base_filename}.txt")
                
                # 기존 파일 확인 - 정확히 같은 설정의 파일만 재사용
                if os.path.exists(reads_bin) and os.path.exists(reads_txt):
                    # 기존 파일의 헤더 정보 확인
                    with open(reads_bin, 'rb') as existing_f:
                        existing_read_length = int(np.fromfile(existing_f, dtype=np.uint64, count=1)[0])
                        if existing_read_length == read_length:
                            print(f"\n동일한 설정의 기존 리드 파일이 존재합니다:")
                            print(f"바이너리 파일: {reads_bin}")
                            print(f"텍스트 파일: {reads_txt}")
                            return reads_bin, reads_txt
                        else:
                            print(f"\n기존 파일의 리드 길이({existing_read_length}bp)가 다릅니다.")
                            print(f"새로운 리드 파일을 생성합니다. (리드 길이: {read_length}bp)")
                
                # 생성될 총 리드 수 계산
                step_size = int(read_length - overlap)
                if step_size <= 0:
                    raise ValueError(f"overlap({overlap})이 read_length({read_length})보다 크거나 같습니다")
                    
                read_count = int((total_length - read_length) // step_size + 1)
                
                print(f"\n=== 리드 생성 정보 ===")
                print(f"시퀀스 길이: {total_length:,}bp")
                print(f"리드 길이: {read_length}bp")
                print(f"리드 간 겹침: {overlap}bp")
                print(f"리드 간격: {step_size}bp")
                print(f"생성할 총 리드 수: {read_count:,}")
                
                # 데이터 디렉토리 설정
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                reads_dir = os.path.join(project_root, 'data', 'reads')
                os.makedirs(reads_dir, exist_ok=True)
                
                # 파일명 생성
                base_filename = f"sequence_{total_length}bp_reads_{read_length}bp"
                reads_bin = os.path.join(reads_dir, f"{base_filename}.bin")
                reads_txt = os.path.join(reads_dir, f"{base_filename}.txt")

                # 헤더 정보 저장 (read_length, read_count)
                with open(reads_bin, 'wb') as bin_f, open(reads_txt, 'w') as txt_f:
                    # 바이너리 파일 헤더
                    header = np.array([read_length, read_count], dtype=np.uint64)
                    header.tofile(bin_f)
                    
                    # 리드 생성 및 저장
                    current_read = 0
                    chunk_reads = []
                    
                    for read in self.generate_reads_stream(f, total_length, read_length, overlap):
                        chunk_reads.append(read)
                        current_read += 1
                        
                        # 진행률 표시
                        if current_read % 1000 == 0:
                            progress = int((current_read/read_count)*100)
                            print(f"\r진행률: {progress}% ({current_read:,}/{read_count:,} 리드)", end="")
                        
                        # 청크 단위로 저장
                        if len(chunk_reads) >= chunk_size:
                            # 바이너리 파일에 저장
                            for chunk_read in chunk_reads:
                                chunk_read.tofile(bin_f)
                            # 텍스트 파일에 저장
                            for chunk_read in chunk_reads:
                                read_str = self.sequence_to_string(chunk_read)
                                txt_f.write(read_str + '\n')
                            chunk_reads = []
                    
                    # 남은 리드 저장
                    if chunk_reads:
                        # 바이너리 파일에 저장
                        for chunk_read in chunk_reads:
                            chunk_read.tofile(bin_f)
                        # 텍스트 파일에 저장
                        for chunk_read in chunk_reads:
                            read_str = self.sequence_to_string(chunk_read)
                            txt_f.write(read_str + '\n')

                print(f"\n\n=== 리드 생성 완료 ===")
                print(f"바이너리 파일: {reads_bin}")
                print(f"텍스트 파일: {reads_txt}")
                return reads_bin, reads_txt
                
        except Exception as e:
            print(f"\n리드 생성 중 오류 발생: {str(e)}")
            return "", ""

    def find_file_pairs(self) -> List[Tuple[str, str, int]]:
        """리드 파일과 해당하는 원본 파일, 리드 길이를 찾아 반환"""
        pairs = []
        
        # reads 디렉토리의 모든 .txt 파일 검색
        read_files = glob.glob(os.path.join(self.reads_dir, '*.bin'))
        
        for read_file in read_files:
            # 파일명에서 시퀀스 길이와 리드 길이 추출
            filename = os.path.basename(read_file)
            if '_reads_' in filename:
                sequence_length = filename.split('_')[0]
                read_length = int(filename.split('_reads_')[1].split('bp')[0])
                
                # 해당하는 원본 파일 찾기
                original_file = os.path.join(self.original_dir, f'sequence_{sequence_length}bp.bin')
                
                if os.path.exists(original_file):
                    pairs.append((read_file, original_file, read_length))
        
        return pairs

if __name__ == "__main__":
    generator = DNASequence()
    bin_path, txt_path = generator.save_sequence(10**4, "test.bin")
    print(f"생성된 파일: {bin_path}")
    bin_path, txt_path = generator.save_reads(bin_path, 80, 50, 10**4)
    print(f"생성된 파일: {bin_path}")
    