from __future__ import annotations

import time

import networkx as nx
import pandas as pd


def run_girvan_newman(graph: nx.Graph, n_clusters: int = 4):
    if graph.number_of_nodes() == 0:
        raise ValueError("Cannot cluster an empty graph")
    if n_clusters < 2:
        raise ValueError("n_clusters must be at least 2")
    start = time.perf_counter()
    iterator = nx.community.girvan_newman(graph)
    communities = None
    for _ in range(n_clusters - 1):
        communities = next(iterator)
    if communities is None:
        raise RuntimeError("Girvan-Newman did not produce a partition")
    communities = [set(community) for community in communities]
    cluster_by_node = {
        node: cluster_id for cluster_id, community in enumerate(communities) for node in community
    }
    assignments = pd.DataFrame(
        {"actor_id": list(cluster_by_node), "cluster_id": list(cluster_by_node.values())}
    ).sort_values("actor_id").reset_index(drop=True)
    metrics = {
        "method": "girvan_newman",
        "n_nodes": graph.number_of_nodes(),
        "n_edges": graph.number_of_edges(),
        "n_clusters": len(communities),
        "modularity": float(nx.community.modularity(graph, communities, weight="weight")),
        "runtime_seconds": time.perf_counter() - start,
        "requested_clusters": n_clusters,
    }
    return assignments, metrics