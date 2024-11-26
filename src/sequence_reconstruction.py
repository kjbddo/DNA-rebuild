import numpy as np
from typing import Generator, List, Dict, Tuple
from collections import defaultdict
import os

def read_reads_streaming(file_path: str, read_length: int = 100, chunk_size: int = 1000) -> Generator[List[np.ndarray], None, None]:
    """바이너리 파일에서 리드를 청크 단위로 읽는 함수"""
    try:
        with open(file_path, 'rb') as f:
            # 헤더 정보 읽기 (read_length, num_reads)
            read_length_from_file = int(np.fromfile(f, dtype=np.uint64, count=1)[0])
            num_reads = int(np.fromfile(f, dtype=np.uint64, count=1)[0])
            
            print(f"\n=== 리드 파일 정보 ===")
            print(f"총 리드 수: {num_reads:,}")
            print(f"리드 길이: {read_length}bp")
            
            chunk = []
            reads_processed = 0
            
            while reads_processed < num_reads:
                # 청크 크기만큼의 데이터를 한 번에 읽기
                chunk_data = f.read(read_length * chunk_size)
                if not chunk_data:
                    break
                
                # 읽은 데이터를 리드 길이만큼 분할
                for i in range(0, len(chunk_data), read_length):
                    read_data = chunk_data[i:i + read_length]
                    if len(read_data) < read_length:
                        break
                    
                    # 바이트 데이터를 numpy 배열로 변환
                    read = np.frombuffer(read_data, dtype=np.uint8)
                    chunk.append(read)
                    reads_processed += 1
                    
                    # 청크가 가득 차면 yield
                    if len(chunk) >= chunk_size:
                        yield chunk
                        chunk = []
                
                if reads_processed % 10000 == 0:
                    print(f"\r진행률: {(reads_processed/num_reads)*100:.1f}% ({reads_processed:,}/{num_reads:,} 리드)", end="")
            
            # 마지막 청크 처리
            if chunk:
                yield chunk
                
            print(f"\n\n처리된 총 리드 수: {reads_processed:,}")
            
    except FileNotFoundError:
        print(f"파일을 찾을 수 없습니다: {file_path}")
        yield []
    except Exception as e:
        print(f"파일 읽기 중 오류 발생: {str(e)}")
        yield []

def evaluate_accuracy_streaming(original_file: str, reconstructed: np.ndarray, chunk_size: int = 1000) -> float:
    """
    재구성된 시퀀스의 정확도를 평가하는 함수
    
    Args:
        original_file (str): 원본 시퀀스 파일 경로
        reconstructed (np.ndarray): 재구성된 시퀀스
        chunk_size (int): 한 번에 비교할 청크의 크기
    
    Returns:
        float: 정확도 (백분율)
    """
    try:
        total_matches = 0
        total_bases = 0
        
        with open(original_file, 'rb') as f:
            while True:
                chunk = f.read(int(chunk_size))
                if not chunk:
                    break
                
                original_chunk = np.frombuffer(chunk, dtype=np.uint8)
                chunk_size = len(original_chunk)
                
                if total_bases + chunk_size > len(reconstructed):
                    # 재구성된 시퀀스가 더 짧은 경우
                    chunk_size = int(len(reconstructed) - total_bases)
                    if chunk_size <= 0:
                        break
                    original_chunk = original_chunk[:chunk_size]
                
                # 현재 청크에서 일치하는 염기 수 계산
                matches = int(np.sum(original_chunk == reconstructed[total_bases:total_bases + chunk_size]))
                total_matches += matches
                total_bases += chunk_size
        
        if total_bases == 0:
            return 0.0
            
        return (total_matches / total_bases) * 100.0
        
    except FileNotFoundError:
        print(f"파일을 찾을 수 없습니다: {original_file}")
        return 0.0
    except Exception as e:
        print(f"정확도 평가 중 오류 발생: {str(e)}")
        return 0.0

def create_kmer_index(reads: List[str], k: int) -> Dict[str, List[int]]:
    """
    k-mer 인덱스를 생성하는 함수
    
    Args:
        reads (List[str]): 리드 목록
        k (int): k-mer 길이
    
    Returns:
        Dict[str, List[int]]: k-mer를 키로, 해당 k-mer가 등장하는 리드의 인덱스 목록을 값으로 하는 사전
    """
    kmer_index = defaultdict(list)
    for i, read in enumerate(reads):
        for j in range(len(read) - k + 1):
            kmer = read[j:j+k]
            kmer_index[kmer].append(i)
    return kmer_index

def find_overlaps(reads: List[str], k: int) -> Dict[int, Dict[int, int]]:
    """
    리드들 간의 중첩을 찾는 함수
    
    Args:
        reads (List[str]): 리드 목록
        k (int): 최소 중첩 길이
    
    Returns:
        Dict[int, Dict[int, int]]: 리드 간의 중첩 정보를 담은 그래프
    """
    overlaps = defaultdict(lambda: defaultdict(int))
    for i, read1 in enumerate(reads):
        for j, read2 in enumerate(reads):
            if i != j:
                for overlap_len in range(len(read1), k-1, -1):
                    if read1.endswith(read2[:overlap_len]):
                        overlaps[i][j] = overlap_len
                        break
    return overlaps

def calculate_coverage(reads: List[str], position: int) -> int:
    """
    특정 위치의 커버리지를 계산하는 함수
    
    Args:
        reads (List[str]): 리드 목록
        position (int): 확인할 위치
    
    Returns:
        int: 해당 위치의 커버리지
    """
    coverage = 0
    for read in reads:
        if position < len(read):
            coverage += 1
    return coverage

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
            # 60bp씩 나누어 저장
            for i in range(0, len(sequence), 60):
                f.write(sequence[i:i+60] + '\n')
        
        print(f"\n=== 재구성된 시퀀스 저장 완료 ===")
        print(f"방법: {method}")
        print(f"시퀀스 길이: {len(reconstructed):,}bp")
        print(f"바이너리 파일: {bin_path}")
        print(f"텍스트 파일: {txt_path}")
        
        return bin_path, txt_path
        
    except Exception as e:
        print(f"시퀀스 저장 중 오류 발생: {str(e)}")
        return "", ""