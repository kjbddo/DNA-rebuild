import numpy as np
from typing import Generator, List, Dict, Tuple
from collections import defaultdict
import os

def read_reads_streaming(file_path: str, read_length, chunk_size) -> Generator[List[np.ndarray], None, None]:
    """바이너리 파일에서 리드를 청크 단위로 읽는 함수"""
    try:
        with open(file_path, 'rb') as f:
            # 헤더 정보 읽기 (read_length, num_reads)
            read_length_from_file = int(np.fromfile(f, dtype=np.uint64, count=1)[0])
            num_reads = int(np.fromfile(f, dtype=np.uint64, count=1)[0])
            
            if read_length_from_file != read_length:
                print(f"경고: 파일의 리드 길이({read_length_from_file})와 입력 리드 길이({read_length})가 다릅니다!")
            
            chunk = []
            reads_processed = 0
            
            while reads_processed < num_reads:
                # 청크 크기만큼의 리드 데이터를 한 번에 읽기
                chunk_data = np.fromfile(f, dtype=np.uint8, count=read_length * chunk_size)
                if len(chunk_data) == 0:
                    break
                
                # 읽은 데이터를 리드 단위로 분할
                num_complete_reads = len(chunk_data) // read_length
                chunk_data = chunk_data[:num_complete_reads * read_length]
                reads = np.array_split(chunk_data, num_complete_reads)
                
                # 청크에 추가
                chunk.extend(reads)
                reads_processed += num_complete_reads
                
                # 청크가 가득 차면 yield
                if len(chunk) >= chunk_size:
                    yield chunk[:chunk_size]
                    chunk = chunk[chunk_size:]
                
                if reads_processed % 10000 == 0:
                    print(f"\r진행률: {(reads_processed/num_reads)*100:.1f}% ({reads_processed:,}/{num_reads:,} 리드)", end="")
            
            # 남은 청크 처리
            if chunk:
                yield chunk
                
    except FileNotFoundError:
        print(f"파일을 찾을 수 없습니다: {file_path}")
        yield []
    except Exception as e:
        print(f"파일 읽기 중 오류 발생: {str(e)}")
        yield []

def save_reconstructed_sequence(reconstructed: np.ndarray, reads_file: str, method: str) -> Tuple[str, str]:
    """재구성된 시퀀스를 바이너리와 텍스트 파일로 저장
    
    Args:
        reconstructed: 재구성된 시퀀스 (numpy array)
        reads_file: 원본 리드 파일 경로
        method: 사용된 재구성 방법 (예: 'brute-force', 'debruijn' 등)
    
    Returns:
        Tuple[str, str]: (바이너리 파일 경로, 텍스트 파일 경로)
    """
    try:
        # 프로젝트 루트 및 저장 디렉토리 설정
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        reconstructed_dir = os.path.join(project_root, 'data', 'reconstructed')
        os.makedirs(reconstructed_dir, exist_ok=True)
        
        # 파일명 생성
        base_name = os.path.splitext(os.path.basename(reads_file))[0]
        reconstructed_base = f"{method}_{base_name}"
        bin_path = os.path.join(reconstructed_dir, f"{reconstructed_base}.bin")
        txt_path = os.path.join(reconstructed_dir, f"{reconstructed_base}.txt")
        
        # 바이너리 파일로 저장
        with open(bin_path, 'wb') as f:
            # 헤더: 시퀀스 길이
            np.array([len(reconstructed)], dtype=np.uint64).tofile(f)
            # 시퀀스 데이터
            reconstructed.tofile(f)
        
        # 텍스트 파일로 저장
        num_to_base = {0: 'A', 1: 'T', 2: 'C', 3: 'G', 4: 'N'}
        with open(txt_path, 'w') as f:
            f.write(f"# Reconstructed DNA Sequence ({method})\n")
            f.write(f"# Length: {len(reconstructed)}bp\n")
            f.write(f"# Original reads file: {reads_file}\n\n")
            
            # 시퀀스를 문자열로 변환하여 저장
            sequence = ''.join(num_to_base[b] for b in reconstructed)
            # 100bp씩 나누어 저장
            for i in range(0, len(sequence), 100):
                f.write(sequence[i:i+100] + '\n')
        
        print(f"\n=== 재구성된 시퀀스 저장 완료 ===")
        print(f"방법: {method}")
        print(f"시퀀스 길이: {len(reconstructed):,}bp")
        print(f"바이너리 파일: {bin_path}")
        print(f"텍스트 파일: {txt_path}")
        
        return bin_path, txt_path
        
    except Exception as e:
        print(f"시퀀스 저장 중 오류 발생: {str(e)}")
        return "", ""