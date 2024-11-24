import numpy as np
from typing import Generator, List, Dict
from collections import defaultdict

def read_reads_streaming(file_path: str, read_length: int = 100, chunk_size: int = 1000) -> Generator[List[np.ndarray], None, None]:
    """바이너리 파일에서 리드를 청크 단위로 읽는 함수"""
    try:
        with open(file_path, 'rb') as f:
            # 헤더 정보 읽기
            header = np.fromfile(f, dtype=np.uint64, count=2)
            stored_read_length = header[0]
            total_reads = header[1]
            
            if stored_read_length != read_length:
                print(f"경고: 저장된 리드 길이({stored_read_length})가 요청된 길이({read_length})와 다릅니다")
            
            chunk = []
            while True:
                read = np.fromfile(f, dtype=np.uint8, count=read_length)
                
                if len(read) < read_length:
                    if chunk:
                        yield chunk
                    break
                
                chunk.append(read)
                
                if len(chunk) >= chunk_size:
                    yield chunk
                    chunk = []
            
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
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                
                original_chunk = np.frombuffer(chunk, dtype=np.uint8)
                chunk_size = len(original_chunk)
                
                if total_bases + chunk_size > len(reconstructed):
                    # 재구성된 시퀀스가 더 짧은 경우
                    chunk_size = len(reconstructed) - total_bases
                    if chunk_size <= 0:
                        break
                    original_chunk = original_chunk[:chunk_size]
                
                # 현재 청크에서 일치하는 염기 수 계산
                matches = np.sum(original_chunk == reconstructed[total_bases:total_bases + chunk_size])
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