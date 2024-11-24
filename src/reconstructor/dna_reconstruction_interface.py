from abc import ABC, abstractmethod
import numpy as np
import os
import sys
from typing import List, Dict, Generator

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.sequence_reconstruction import evaluate_accuracy_streaming

class DNAReconstructor(ABC):
    def __init__(self, k: int = 31, min_coverage: int = 2, chunk_size: int = 10**6, read_length: int = 100):
        self.k = k
        self.min_coverage = min_coverage
        self.chunk_size = chunk_size
        self.read_length = read_length
    
    @abstractmethod
    def reconstruct(self, reads_file: str) -> np.ndarray:
        pass
    
    def validate(self, original_file: str, reconstructed: np.ndarray) -> float:
        """정확도 검증"""
        return evaluate_accuracy_streaming(original_file, reconstructed)
