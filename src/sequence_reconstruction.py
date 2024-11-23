from collections import defaultdict
from typing import List, Dict, Tuple, Generator
import numpy as np
import os

def read_reads_streaming(filename: str, chunk_size: int = 1000) -> Generator[List[np.ndarray], None, Tuple[int, int]]:
    """리드를 청크 단위로 읽어오는 제너레이터"""
    with open(filename, 'rb') as f:
        # 메타데이터 읽기
        read_count, read_length = np.fromfile(f, dtype=np.uint32, count=2)
        
        reads = []
        for _ in range(read_count):
            read = np.fromfile(f, dtype=np.uint8, count=read_length)
            if len(read) == read_length:
                reads.append(read)
                
                if len(reads) >= chunk_size:
                    yield reads
                    reads = []
        
        if reads:  # 남은 리드들 처리
            yield reads
            
    return read_count, read_length

def construct_debruijn_graph_streaming(reads_file: str, k: int, min_coverage: int = 2) -> Dict:
    kmer_freq = defaultdict(int)
    edges = defaultdict(lambda: defaultdict(int))
    
    # DNA 염기 매핑
    base_map = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
    
    print("\nk-mer 빈도 계산 중...")
    total_kmers = 0
    
    for reads_chunk in read_reads_streaming(reads_file):
        for read in reads_chunk:
            # 숫자를 DNA 염기로 변환
            read_str = ''.join(base_map[b] for b in read)
            for i in range(len(read_str) - k + 1):
                kmer = read_str[i:i+k]
                kmer_freq[kmer] += 1
                total_kmers += 1
    
    # 그래프 구성
    for reads_chunk in read_reads_streaming(reads_file):
        for read in reads_chunk:
            read_str = ''.join(base_map[b] for b in read)
            for i in range(len(read_str) - k + 1):
                kmer = read_str[i:i+k]
                if kmer_freq[kmer] >= min_coverage:
                    prefix = kmer[:-1]
                    suffix = kmer[1:]
                    edges[prefix][suffix] += kmer_freq[kmer]  # 빈도를 가중치로 사용
    
    return edges

def simplify_graph(graph: Dict) -> Dict:
    """그래프 단순화 - 보수적인 접근
    - 짧은 데드엔드만 제거
    - 매우 유사한 버블만 병합
    """
    def remove_dead_ends(g, max_length=20):  # max_length를 더 작게 조정
        """짧은 데드엔드 경로만 제거"""
        changed = True
        while changed:
            changed = False
            dead_ends = []
            for node in g:
                # 진출 차수가 0이고 진입 차수가 1인 노드만 처리
                if len(g[node]) == 0:  # 진출 차수 0
                    in_degree = sum(1 for n in g if node in g[n])
                    if in_degree == 1:  # 진입 차수 1
                        path_length = 1
                        current = node
                        while path_length < max_length:
                            prev_nodes = [n for n in g if current in g[n]]
                            if len(prev_nodes) != 1:
                                break
                            current = prev_nodes[0]
                            path_length += 1
                        
                        if path_length < max_length:
                            dead_ends.append(node)
                            changed = True
            
            for node in dead_ends:
                del g[node]
        return g
    
    def merge_bubbles(g, max_length=30, similarity_threshold=0.9):  # 더 엄격한 기준 적용
        """매우 유사한 버블만 병합"""
        def find_bubble(start_node):
            paths = []
            def dfs(node, path, length):
                if length > max_length:
                    return
                if not g[node]:  # 끝점
                    paths.append(path)
                    return
                for next_node in g[node]:
                    dfs(next_node, path + [next_node], length + 1)
            
            dfs(start_node, [start_node], 0)
            return paths
        
        def calculate_similarity(path1, path2):
            """두 경로의 유사도 계산"""
            if abs(len(path1) - len(path2)) > 2:  # 길이 차이가 크면 다른 경로로 취급
                return 0.0
            
            # 경로의 염기 서열 유사도 계산
            matches = sum(1 for x, y in zip(path1, path2) if x == y)
            total = max(len(path1), len(path2))
            return matches / total
        
        changed = True
        iterations = 0
        max_iterations = 100  # 무한 루프 방지
        
        while changed and iterations < max_iterations:
            changed = False
            iterations += 1
            
            for node in list(g.keys()):
                if len(g[node]) > 1:  # 분기점
                    paths = find_bubble(node)
                    if len(paths) > 1:
                        # 가장 긴 경로를 기준으로 유사한 경로만 병합
                        main_path = max(paths, key=len)
                        for path in paths:
                            if path != main_path:
                                similarity = calculate_similarity(path, main_path)
                                if similarity >= similarity_threshold:
                                    # 매우 유사한 경로만 제거
                                    for n in path[1:-1]:
                                        if n in g:
                                            del g[n]
                                    changed = True
        return g
    
    # 그래프 단순화 적용 (순서 중요)
    print("그래프 단순화 시작...")
    initial_nodes = len(graph)
    
    graph = remove_dead_ends(graph)
    after_dead_ends = len(graph)
    print(f"데드엔드 제거 후 노드 수: {after_dead_ends} (제거됨: {initial_nodes - after_dead_ends})")
    
    graph = merge_bubbles(graph)
    final_nodes = len(graph)
    print(f"버블 병합 후 노드 수: {final_nodes} (제거됨: {after_dead_ends - final_nodes})")
    
    return graph

def reconstruct_sequence_streaming(reads_file: str, k: int, min_coverage: int = 2) -> np.ndarray:
    """향상된 시퀀스 재구성"""
    # DNA 염기 매핑 (문자 -> 숫자)
    base_to_num = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    
    print("De Bruijn 그래프 구성 중...")
    graph = construct_debruijn_graph_streaming(reads_file, k, min_coverage)
    print(f"초기 그래프 생성 완료 (노드 수: {len(graph)})")
    
    print("그래프 단순화 중...")
    graph = simplify_graph(graph)
    print(f"단순화 후 노드 수: {len(graph)}")
    
    print("오일러 경로 찾는 중...")
    path = find_eulerian_path(graph)
    print("경로 찾기 완료")
    
    if not path:
        print("경로를 찾을 수 없습니다!")
        return np.array([], dtype=np.uint8)
    
    # 경로를 시퀀스로 변환
    print("시퀀스 재구성 중...")
    # 첫 번째 노드의 모든 문자를 숫자로 변환
    reconstructed = [base_to_num[x] for x in path[0]]
    
    # 이후 노드들의 마지막 문자만 추가
    for node in path[1:]:
        reconstructed.append(base_to_num[node[-1]])
    
    print(f"재구성된 시퀀스 길이: {len(reconstructed)}")
    return np.array(reconstructed, dtype=np.uint8)

def evaluate_accuracy_streaming(original_file: str, reconstructed: np.ndarray, 
                             chunk_size: int = 10**6) -> float:
    """청크 단위로 정확도 평가"""
    total_bases = 0
    matching_bases = 0
    
    with open(original_file, 'rb') as f:
        # 원본 시퀀스 길이 읽기
        sequence_length = np.fromfile(f, dtype=np.uint64, count=1)[0]
        min_length = min(sequence_length, len(reconstructed))
        
        for i in range(0, min_length, chunk_size):
            current_chunk_size = min(chunk_size, min_length - i)
            original_chunk = np.fromfile(f, dtype=np.uint8, count=current_chunk_size)
            reconstructed_chunk = reconstructed[i:i+current_chunk_size]
            
            matching_bases += sum(1 for x, y in zip(original_chunk, reconstructed_chunk) if x == y)
            total_bases += len(original_chunk)
            
            if i % (chunk_size * 10) == 0:
                print(f"비교 진행률: {i/min_length*100:.1f}%")
    
    return (matching_bases / total_bases) * 100 if total_bases > 0 else 0.0

def find_eulerian_path(graph: Dict) -> List[str]:
    """개선된 오일러 경로 찾기"""
    if not graph:
        return []
    
    # 시작점 찾기
    def find_start_node():
        in_degree = defaultdict(int)
        out_degree = defaultdict(int)
        
        # 진입/진출 차수 계산
        for node in graph:
            out_degree[node] = sum(graph[node].values())
            for next_node, count in graph[node].items():
                in_degree[next_node] += count
        
        # 시작점 후보 찾기
        start_candidates = []
        for node in graph:
            out_deg = out_degree[node]
            in_deg = in_degree[node]
            
            if out_deg > in_deg:  # 진출차수가 더 큰 노드
                return node
            elif out_deg > 0:  # 진출차수가 있는 노드
                start_candidates.append(node)
        
        return start_candidates[0] if start_candidates else next(iter(graph))
    
    # Hierholzer 알고리즘
    def find_path(start):
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
                if not current_graph[current]:
                    del current_graph[current]
                
                stack.append(next_node)
            else:
                path.append(stack.pop())
        
        return path[::-1]
    
    start_node = find_start_node()
    path = find_path(start_node)
    
    # 경로 검증
    if len(path) < 2:
        print("경고: 경로가 너무 짧습니다!")
        return []
    
    # 모든 엣지를 사용했는지 확인
    unused_edges = sum(sum(edges.values()) for edges in graph.values())
    if unused_edges > 0:
        print(f"경고: {unused_edges}개의 미사용 엣지가 있습니다!")
    
    return path

def validate_reconstruction(original_file: str, reconstructed: np.ndarray, 
                          reads_file: str, k: int) -> Tuple[float, float]:
    """재구성된 시퀀스 검증
    Returns:
        Tuple[float, float]: (정확도, 리드 매핑률)
    """
    # 정확도 계산
    accuracy = evaluate_accuracy_streaming(original_file, reconstructed)
    
    # 리드 매핑률 계산
    mapped_reads = 0
    total_reads = 0
    reconstructed_kmers = set()
    
    # 재구성된 시퀀스의 k-mer 집합 생성
    for i in range(len(reconstructed) - k + 1):
        reconstructed_kmers.add(tuple(reconstructed[i:i+k]))
    
    # 리드 매핑 확인
    for reads_chunk in read_reads_streaming(reads_file):
        for read in reads_chunk:
            total_reads += 1
            read_mapped = False
            for i in range(len(read) - k + 1):
                if tuple(read[i:i+k]) in reconstructed_kmers:
                    read_mapped = True
                    break
            if read_mapped:
                mapped_reads += 1
    
    mapping_rate = (mapped_reads / total_reads * 100) if total_reads > 0 else 0
    
    print(f"시퀀스 재구성 검증 결과:")
    print(f"- 정확도: {accuracy:.2f}%")
    print(f"- 리드 매핑률: {mapping_rate:.2f}%")
    
    return accuracy, mapping_rate