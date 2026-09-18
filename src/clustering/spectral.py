from __future__ import annotations

import time

import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import SpectralClustering


def run_spectral(graph: nx.Graph, n_clusters: int, seed: int = 42):
    if graph.number_of_nodes() == 0:
        raise ValueError("Cannot cluster an empty graph")
    if n_clusters < 2 or n_clusters > graph.number_of_nodes():
        raise ValueError("n_clusters must be between 2 and the number of nodes")
    nodes = sorted(graph.nodes(), key=str)
    adjacency = nx.to_scipy_sparse_array(
        graph, nodelist=nodes, weight="weight", format="csr"
    )
    adjacency = sparse.csr_matrix(adjacency, dtype=np.float64)
    adjacency.indices = adjacency.indices.astype(np.int32, copy=False)
    adjacency.indptr = adjacency.indptr.astype(np.int32, copy=False)
    start = time.perf_counter()
    model = SpectralClustering(
        n_clusters=n_clusters,
        affinity="precomputed",
        assign_labels="kmeans",
        random_state=seed,
        n_init=10,
    )
    labels = model.fit_predict(adjacency)
    assignments = pd.DataFrame({"actor_id": nodes, "cluster_id": labels})
    metrics = {
        "method": "spectral",
        "n_nodes": graph.number_of_nodes(),
        "n_edges": graph.number_of_edges(),
        "n_clusters": int(np.unique(labels).size),
        "runtime_seconds": time.perf_counter() - start,
        "seed": seed,
        "requested_clusters": n_clusters,
    }
    return assignments, metrics