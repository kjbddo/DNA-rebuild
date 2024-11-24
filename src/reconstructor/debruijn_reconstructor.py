from collections import defaultdict
from typing import Dict, List, Tuple
import numpy as np
from src.reconstructor.dna_reconstruction_interface import DNAReconstructor
from src.sequence_reconstruction import read_reads_streaming

class DeBruijnReconstructor(DNAReconstructor):
    def __init__(self, k: int = 31, min_coverage: int = 2, chunk_size: int = 10**6, read_length: int = 100):
        super().__init__(k, min_coverage, chunk_size, read_length)
        self.base_to_num = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        self.num_to_base = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
    
    def reconstruct(self, reads_file: str) -> np.ndarray:
        """DBG 알고리즘을 사용한 시퀀스 재구성"""
        print("De Bruijn 그래프 구성 중...")
        graph = self._construct_graph(reads_file)
        print(f"초기 그래프 생성 완료 (노드 수: {len(graph)})")
        
        print("그래프 단순화 중...")
        graph = self._simplify_graph(graph)
        print(f"단순화 후 노드 수: {len(graph)}")
        
        print("오일러 경로 찾는 중...")
        path = self._find_eulerian_path(graph)
        print("경로 찾기 완료")
        
        if not path:
            print("경로를 찾을 수 없습니다!")
            return np.array([], dtype=np.uint8)
        
        return self._path_to_sequence(path)
    
    def _construct_graph(self, reads_file: str) -> Dict:
        """De Bruijn 그래프 구성"""
        kmer_freq = defaultdict(int)
        edges = defaultdict(lambda: defaultdict(int))
        
        # 첫 번째 패스: k-mer 빈도 계산
        print("\nk-mer 빈도 계산 중...")
        for reads_chunk in read_reads_streaming(reads_file, read_length=self.read_length):
            for read in reads_chunk:
                read_str = ''.join(self.num_to_base[b] for b in read)
                for i in range(len(read_str) - self.k + 1):
                    kmer = read_str[i:i+self.k]
                    kmer_freq[kmer] += 1
        
        # 두 번째 패스: 그래프 구성
        for reads_chunk in read_reads_streaming(reads_file):
            for read in reads_chunk:
                read_str = ''.join(self.num_to_base[b] for b in read)
                for i in range(len(read_str) - self.k + 1):
                    kmer = read_str[i:i+self.k]
                    if kmer_freq[kmer] >= self.min_coverage:
                        prefix = kmer[:-1]
                        suffix = kmer[1:]
                        edges[prefix][suffix] += kmer_freq[kmer]
        
        return edges
    
    def _simplify_graph(self, graph: Dict) -> Dict:
        """그래프 단순화"""
        def remove_dead_ends(g: Dict, max_length: int = 20) -> Dict:
            changed = True
            while changed:
                changed = False
                dead_ends = []
                for node in g:
                    if len(g[node]) == 0:  # 진출 엣지 없음
                        in_degree = sum(1 for n in g if node in g[n])
                        if in_degree == 1:  # 진입 엣지가 하나인 경우만
                            dead_ends.append(node)
                            changed = True
                
                for node in dead_ends:
                    del g[node]
            return g
        
        def merge_bubbles(g: Dict, similarity_threshold: float = 0.9) -> Dict:
            changed = True
            while changed:
                changed = False
                for node in list(g.keys()):
                    if len(g[node]) > 1:  # 분기점
                        paths = self._find_bubble_paths(g, node)
                        if len(paths) > 1:
                            # 유사도가 높은 경로 병합
                            main_path = max(paths, key=len)
                            for path in paths:
                                if path != main_path:
                                    similarity = self._calculate_path_similarity(path, main_path)
                                    if similarity >= similarity_threshold:
                                        for n in path[1:-1]:
                                            if n in g:
                                                del g[n]
                                        changed = True
            return g
        
        # 그래프 단순화 적용
        graph = remove_dead_ends(graph)
        graph = merge_bubbles(graph)
        return graph
    
    def _find_eulerian_path(self, graph: Dict) -> List[str]:
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