from collections import defaultdict
from typing import Dict, List, Tuple
import numpy as np
from src.reconstructor.dna_reconstruction_interface import DNAReconstructor
from src.sequence_reconstruction import read_reads_streaming

class DeBruijnReconstructor(DNAReconstructor):
    def __init__(self, k: int = 31, min_coverage: int = 2, chunk_size: int = 10**6, read_length: int = 100):
        super().__init__(k, min_coverage, chunk_size, read_length)
        self.base_to_num = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
        self.num_to_base = {0: 'A', 1: 'T', 2: 'C', 3: 'G'}
    
    def reconstruct(self, reads_file: str) -> np.ndarray:
        """De Bruijn 그래프를 사용하여 DNA 시퀀스 재구성"""
        reads = []
        print("리드 데이터 로딩 중...")
        
        # 바이너리 파일에서 리드 데이터 로드
        for reads_chunk in read_reads_streaming(reads_file, read_length=self.read_length):
            reads.extend(reads_chunk)
        
        print(f"총 {len(reads)}개의 리드 로드 완료")
        
        # 리드를 문자열로 변환
        reads_str = [self._convert_read_to_string(read) for read in reads]
        
        # De Bruijn 그래프 구성
        graph = self._build_graph(reads_str)
        
        # 오일러 경로 찾기
        path = self._find_euler_path(graph)
        
        # 경로를 시퀀스로 변환
        sequence = self._path_to_sequence(path)
        
        # numpy 배열의 길이로 검사
        if len(sequence) == 0:
            print("시퀀스 재구성 실패!")
            return np.array([], dtype=np.uint8)
        
        print("시퀀스 재구성 완료")
        return np.array([self.base_to_num[b] for b in sequence], dtype=np.uint8)
    
    def _convert_read_to_string(self, read: np.ndarray) -> str:
        """숫자 배열을 염기서열 문자열로 변환"""
        return ''.join(self.num_to_base[b] for b in read)
    
    def _build_graph(self, reads_str: List[str]) -> Dict:
        """De Bruijn 그래프 구성"""
        kmer_freq = defaultdict(int)
        edges = defaultdict(lambda: defaultdict(int))
        
        # 첫 번째 패스: k-mer 빈도 계산
        print("\nk-mer 빈도 계산 중...")
        for read in reads_str:
            for i in range(len(read) - self.k + 1):
                kmer = read[i:i+self.k]
                kmer_freq[kmer] += 1
        
        # 두 번째 패스: 그래프 구성
        for read in reads_str:
            for i in range(len(read) - self.k + 1):
                kmer = read[i:i+self.k]
                if kmer_freq[kmer] >= self.min_coverage:
                    prefix = kmer[:-1]
                    suffix = kmer[1:]
                    edges[prefix][suffix] += kmer_freq[kmer]
        
        return edges
    
    def _find_euler_path(self, graph: Dict) -> List[str]:
        """오일러 경로 찾기"""
        if not graph:
            return []
        
        def find_start_node():
            in_degree = defaultdict(int)
            out_degree = defaultdict(int)
            
            for node in graph:
                out_degree[node] = sum(graph[node].values())
                for next_node, count in graph[node].items():
                    in_degree[next_node] += count
            
            # 시작점 선택: 진출차수 > 진입차수
            for node in graph:
                if out_degree[node] > in_degree[node]:
                    return node
            return next(iter(graph))
        
        def find_path(start: str) -> List[str]:
            stack = [start]
            path = []
            current_graph = {node: defaultdict(int, edges) for node, edges in graph.items()}
            
            while stack:
                current = stack[-1]
                if current in current_graph and current_graph[current]:
                    # 가중치가 가장 높은 엣지 선택
                    next_node = max(current_graph[current].items(), key=lambda x: x[1])[0]
                    current_graph[current][next_node] -= 1
                    if current_graph[current][next_node] == 0:
                        del current_graph[current][next_node]
                    stack.append(next_node)
                else:
                    path.append(stack.pop())
            
            return path[::-1]
        
        return find_path(find_start_node())
    
    def _path_to_sequence(self, path: List[str]) -> np.ndarray:
        """경로를 시퀀스로 변환"""
        if not path:
            return np.array([], dtype=np.uint8)
        
        # 첫 번째 노드의 모든 문자를 포함
        reconstructed = [self.base_to_num[x] for x in path[0]]
        
        # 이후 노드들의 마지막 문자만 추가
        for node in path[1:]:
            reconstructed.append(self.base_to_num[node[-1]])
        
        return np.array(reconstructed, dtype=np.uint8)
    
    def _find_bubble_paths(self, graph: Dict, start_node: str, max_length: int = 30) -> List[List[str]]:
        """버블 경로 찾기"""
        paths = []
        def dfs(node: str, path: List[str], length: int):
            if length > max_length:
                return
            if not graph[node]:
                paths.append(path)
                return
            for next_node in graph[node]:
                dfs(next_node, path + [next_node], length + 1)
        
        dfs(start_node, [start_node], 0)
        return paths
    
    def _calculate_path_similarity(self, path1: List[str], path2: List[str]) -> float:
        """두 경로의 유사도 계산"""
        if abs(len(path1) - len(path2)) > 2:
            return 0.0
        matches = sum(1 for x, y in zip(path1, path2) if x == y)
        total = max(len(path1), len(path2))
        return matches / total 