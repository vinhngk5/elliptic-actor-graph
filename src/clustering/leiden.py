from __future__ import annotations

import time

import igraph as ig
import leidenalg
import networkx as nx
import pandas as pd


def run_leiden(graph: nx.Graph, seed: int = 42, resolution: float = 1.0):
    if graph.number_of_nodes() == 0:
        raise ValueError("Cannot cluster an empty graph")
    nodes = list(graph.nodes())
    node_to_index = {node: index for index, node in enumerate(nodes)}
    edge_list = [(node_to_index[u], node_to_index[v]) for u, v in graph.edges()]
    weights = [graph[u][v].get("weight", 1.0) for u, v in graph.edges()]
    ig_graph = ig.Graph(n=len(nodes), edges=edge_list, directed=False)
    ig_graph.es["weight"] = weights

    start = time.perf_counter()
    partition = leidenalg.find_partition(
        ig_graph,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=resolution,
        seed=seed,
    )
    cluster_by_node = {
        nodes[index]: cluster_id
        for cluster_id, community in enumerate(partition)
        for index in community
    }
    assignments = pd.DataFrame(
        {"actor_id": list(cluster_by_node), "cluster_id": list(cluster_by_node.values())}
    ).sort_values("actor_id").reset_index(drop=True)
    communities = [set(nodes[index] for index in community) for community in partition]
    metrics = {
        "method": "leiden",
        "n_nodes": graph.number_of_nodes(),
        "n_edges": graph.number_of_edges(),
        "n_clusters": assignments["cluster_id"].nunique(),
        "modularity": float(nx.community.modularity(graph, communities, weight="weight")),
        "runtime_seconds": time.perf_counter() - start,
        "seed": seed,
        "resolution": resolution,
    }
    return assignments, metrics