from __future__ import annotations

import time
from collections import Counter

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD


def _random_walk(graph, start, length, rng):
    walk = [start]
    current = start
    for _ in range(length - 1):
        neighbors = list(graph.neighbors(current))
        if not neighbors:
            break
        current = neighbors[int(rng.integers(len(neighbors)))]
        walk.append(current)
    return walk


def run_node2vec_kmeans(
    graph: nx.Graph,
    n_clusters: int = 8,
    dimensions: int = 32,
    walk_length: int = 20,
    walks_per_node: int = 2,
    window_size: int = 5,
    seed: int = 42,
):
    if graph.number_of_nodes() == 0:
        raise ValueError("Cannot embed an empty graph")
    nodes = sorted(graph.nodes(), key=str)
    if n_clusters < 2 or n_clusters > len(nodes):
        raise ValueError("n_clusters must be between 2 and the number of nodes")
    start = time.perf_counter()
    rng = np.random.default_rng(seed)
    cooccurrence = Counter()
    for node in nodes:
        for _ in range(walks_per_node):
            walk = _random_walk(graph, node, walk_length, rng)
            for index, source in enumerate(walk):
                for context in walk[max(0, index - window_size): index + window_size + 1]:
                    if source != context:
                        cooccurrence[(source, context)] += 1
    matrix = np.zeros((len(nodes), len(nodes)), dtype=np.float32)
    index_by_node = {node: index for index, node in enumerate(nodes)}
    for (source, context), count in cooccurrence.items():
        matrix[index_by_node[source], index_by_node[context]] = count
    n_components = min(dimensions, max(1, len(nodes) - 1), matrix.shape[1])
    embedding = TruncatedSVD(n_components=n_components, random_state=seed).fit_transform(matrix)
    labels = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit_predict(embedding)
    assignments = pd.DataFrame({"actor_id": nodes, "cluster_id": labels})
    metrics = {
        "method": "node2vec_kmeans",
        "n_nodes": len(nodes),
        "n_edges": graph.number_of_edges(),
        "n_clusters": n_clusters,
        "embedding_dimensions": n_components,
        "runtime_seconds": time.perf_counter() - start,
        "seed": seed,
    }
    return assignments, embedding, metrics