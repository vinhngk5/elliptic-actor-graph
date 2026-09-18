from __future__ import annotations

import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans


def run_gae_kmeans(
    graph: nx.Graph,
    features: np.ndarray,
    node_order: list,
    n_clusters: int = 8,
    embedding_dimensions: int = 32,
    epochs: int = 40,
    seed: int = 42,
    checkpoint_path: str | None = None,
):
    if len(node_order) != features.shape[0] or set(node_order) != set(graph.nodes()):
        raise ValueError("features, node_order, and graph nodes must be aligned")
    if len(node_order) > 5000:
        raise ValueError("Dense GAE runner is limited to 5000 nodes; use sparse training for full graph")
    torch.manual_seed(seed)
    np.random.seed(seed)
    start = time.perf_counter()
    node_index = {node: index for index, node in enumerate(node_order)}
    adjacency = nx.to_numpy_array(graph, nodelist=node_order, weight="weight", dtype=np.float32)
    adjacency = (adjacency > 0).astype(np.float32)
    np.fill_diagonal(adjacency, 1.0)
    degree = adjacency.sum(axis=1, keepdims=True)
    normalized = adjacency / np.sqrt(degree @ degree.T)
    x = torch.tensor(features, dtype=torch.float32)
    target = torch.tensor(adjacency, dtype=torch.float32)
    propagation = torch.tensor(normalized, dtype=torch.float32)
    encoder = torch.nn.Sequential(
        torch.nn.Linear(features.shape[1], embedding_dimensions),
        torch.nn.ReLU(),
    )
    optimizer = torch.optim.Adam(encoder.parameters(), lr=0.01)
    loss_history = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        embedding = propagation @ encoder(x)
        reconstruction = torch.sigmoid(embedding @ embedding.T)
        loss = torch.nn.functional.binary_cross_entropy(reconstruction, target)
        loss.backward()
        optimizer.step()
        loss_history.append(float(loss.detach().cpu()))
    with torch.no_grad():
        embedding = (propagation @ encoder(x)).cpu().numpy()
    labels = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit_predict(embedding)
    assignments = pd.DataFrame({"actor_id": node_order, "cluster_id": labels})
    metrics = {
        "method": "gae_kmeans",
        "n_nodes": len(node_order),
        "n_edges": graph.number_of_edges(),
        "n_clusters": n_clusters,
        "embedding_dimensions": embedding_dimensions,
        "epochs": epochs,
        "runtime_seconds": time.perf_counter() - start,
        "seed": seed,
        "final_loss": loss_history[-1],
        "loss_history": loss_history,
    }
    if checkpoint_path is not None:
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": encoder.state_dict(),
                "embedding_dimensions": embedding_dimensions,
                "feature_dimensions": int(features.shape[1]),
                "seed": seed,
                "loss_history": loss_history,
            },
            checkpoint_path,
        )
    return assignments, embedding, metrics