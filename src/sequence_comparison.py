import numpy as np
import os
from typing import Tuple

def read_sequence_from_bin(file_path: str) -> np.ndarray:
    """바이너리 파일에서 시퀀스 읽기"""
    try:
        with open(file_path, 'rb') as f:
            # 헤더에서 시퀀스 길이 읽기
            sequence_length = int(np.fromfile(f, dtype=np.uint64, count=1)[0])
            # 시퀀스 데이터 읽기
            sequence = np.fromfile(f, dtype=np.uint8)
            return sequence
    except FileNotFoundError:
        print(f"파일을 찾을 수 없습니다: {file_path}")
        return np.array([])
    except Exception as e:
        print(f"파일 읽기 오류: {str(e)}")
        return np.array([])

def calculate_accuracy(original: np.ndarray, reconstructed: np.ndarray) -> Tuple[float, int, int]:
    """두 시퀀스의 일치율 계산"""
    # 더 짧은 시퀀스 길이까지만 비교
    compare_length = min(len(original), len(reconstructed))
    
    if compare_length == 0:
        return 0.0, 0, 0
        
    # 일치하는 염기 수 계산
    matches = np.sum(original[:compare_length] == reconstructed[:compare_length])
    
    accuracy = (matches / compare_length) * 100
    return accuracy, matches, compare_length

def compare_sequences(original_file: str, reconstructed_file: str) -> None:
    """원본 시퀀스와 재구성된 시퀀스 비교"""
    print("\n=== 시퀀스 비교 시작 ===")
    
    # 파일 읽기
    original = read_sequence_from_bin(original_file)
    reconstructed = read_sequence_from_bin(reconstructed_file)
    
    if len(original) == 0 or len(reconstructed) == 0:
        print("시퀀스 비교 실패: 파일을 읽을 수 없습니다.")
        return
        
    # 시퀀스 길이 출력
    print(f"\n원본 시퀀스 길이: {len(original):,}bp")
    print(f"재구성 시퀀스 길이: {len(reconstructed):,}bp")
    
    # 일치율 계산
    accuracy, matches, compare_length = calculate_accuracy(original, reconstructed)
    
    # 결과 출력
    print(f"\n=== 비교 결과 ===")
    print(f"비교 길이: {compare_length:,}bp")
    print(f"일치하는 염기 수: {matches:,}")
    print(f"일치율: {accuracy:.2f}%")
    
    # 길이 차이가 있는 경우 추가 정보 출력
    length_diff = abs(len(original) - len(reconstructed))
    if length_diff > 0:
        print(f"\n주의: 시퀀스 길이 차이 {length_diff:,}bp")
        if len(original) > len(reconstructed):
            print("재구성된 시퀀스가 더 짧습니다.")
        else:
            print("재구성된 시퀀스가 더 깁니다.")

def main():
    """메인 함수"""
    # 프로젝트 루트 디렉토리 설정
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 파일 경로 설정
    original_dir = os.path.join(project_root, 'data', 'original')
    reconstructed_dir = os.path.join(project_root, 'data', 'reconstructed')
    
    # 비교할 파일 선택
    original_file = input("원본 시퀀스 파일명을 입력하세요: ")
    reconstructed_file = input("재구성된 시퀀스 파일명을 입력하세요: ")
    
    original_path = os.path.join(original_dir, original_file)
    reconstructed_path = os.path.join(reconstructed_dir, reconstructed_file)
    
    # 시퀀스 비교 실행
    compare_sequences(original_path, reconstructed_path)

if __name__ == "__main__":
    main() 