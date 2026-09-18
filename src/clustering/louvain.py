from __future__ import annotations

import time

import networkx as nx
import pandas as pd


def run_louvain(
    graph: nx.Graph,
    seed: int = 42,
    resolution: float = 1.0,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Run deterministic Louvain community detection on an actor graph."""
    if graph.number_of_nodes() == 0:
        raise ValueError("Cannot cluster an empty graph")
    if resolution <= 0:
        raise ValueError("resolution must be positive")

    start = time.perf_counter()
    communities = nx.community.louvain_communities(
        graph,
        weight="weight",
        resolution=resolution,
        seed=seed,
    )
    ordered_communities = sorted(
        communities,
        key=lambda community: min(str(node) for node in community),
    )
    cluster_by_node = {
        node: cluster_id
        for cluster_id, community in enumerate(ordered_communities)
        for node in community
    }
    assignments = pd.DataFrame(
        {
            "actor_id": list(cluster_by_node),
            "cluster_id": list(cluster_by_node.values()),
        }
    ).sort_values("actor_id").reset_index(drop=True)
    runtime = time.perf_counter() - start
    metrics = {
        "method": "louvain",
        "n_nodes": graph.number_of_nodes(),
        "n_edges": graph.number_of_edges(),
        "n_clusters": assignments["cluster_id"].nunique(),
        "modularity": float(nx.community.modularity(graph, ordered_communities, weight="weight")),
        "runtime_seconds": runtime,
        "seed": seed,
        "resolution": resolution,
    }
    return assignments, metrics