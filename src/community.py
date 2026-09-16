"""Memory-conscious Leiden community detection for the actor graph."""

from __future__ import annotations

from pathlib import Path

import igraph as ig
import leidenalg
import numpy as np
import pandas as pd


def load_integer_edges(
    edge_path: Path,
    addresses: pd.Index,
    chunksize: int = 250_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert string actor edges to compact integer arrays chunk by chunk."""
    address_to_id = pd.Series(np.arange(len(addresses), dtype=np.int32), index=addresses.astype(str))
    edge_parts: list[np.ndarray] = []
    for chunk in pd.read_csv(
        edge_path,
        usecols=["input_address", "output_address"],
        dtype=str,
        chunksize=chunksize,
    ):
        source = chunk["input_address"].map(address_to_id)
        target = chunk["output_address"].map(address_to_id)
        valid = source.notna() & target.notna() & (source != target)
        edge_parts.append(
            np.column_stack(
                [
                    source[valid].to_numpy(dtype=np.int32),
                    target[valid].to_numpy(dtype=np.int32),
                ]
            )
        )
    edges = np.concatenate(edge_parts, axis=0)
    return edges[:, 0], edges[:, 1]


def fit_leiden(
    edge_path: Path,
    addresses: pd.Index,
    resolution: float = 1.0,
    chunksize: int = 250_000,
    seed: int = 42,
) -> tuple[pd.Series, dict[str, float | int]]:
    """Fit weighted undirected Leiden communities on all non-self-loop edges."""
    source, target = load_integer_edges(edge_path, addresses, chunksize)
    edges = np.column_stack([source, target])
    unique_edges, weights = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
    graph = ig.Graph(n=len(addresses), edges=unique_edges.tolist(), directed=False)
    partition = leidenalg.find_partition(
        graph,
        leidenalg.RBConfigurationVertexPartition,
        weights=weights.tolist(),
        resolution_parameter=resolution,
        seed=seed,
    )
    memberships = pd.Series(
        partition.membership,
        index=addresses,
        name="community",
    )
    summary = {
        "input_edges_without_self_loops": int(len(source)),
        "unique_undirected_edges": int(len(unique_edges)),
        "communities": int(len(partition)),
        "modularity": float(partition.modularity),
        "largest_community": int(memberships.value_counts().iloc[0]),
        "seed": seed,
    }
    return memberships, summary