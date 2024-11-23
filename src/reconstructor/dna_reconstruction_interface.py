from abc import ABC, abstractmethod
import numpy as np
from typing import List, Dict, Generator

from src.sequence_reconstruction import evaluate_accuracy_streaming

class DNAReconstructor(ABC):
    def __init__(self, k: int = 31, min_coverage: int = 2, chunk_size: int = 10**6):
        self.k = k
        self.min_coverage = min_coverage
        self.chunk_size = chunk_size
    
    @abstractmethod
    def reconstruct(self, reads_file: str) -> np.ndarray:
        pass
    
    def validate(self, original_file: str, reconstructed: np.ndarray) -> float:
        """정확도 검증"""
        return evaluate_accuracy_streaming(original_file, reconstructed)
