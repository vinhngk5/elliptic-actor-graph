"""Memory-bounded renderers for the complete actor graph."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def render_full_adjacency_density(
    edge_path: Path,
    graph_features: pd.DataFrame,
    output_path: Path,
    image_size: tuple[int, int] = (1600, 1600),
    chunksize: int = 250_000,
) -> dict[str, int]:
    """Render every edge into a degree-ordered adjacency-density image.

    The graph is represented as a density matrix rather than a force-directed
    drawing. This keeps memory bounded and makes hubs and dense source-target
    regions visible even for millions of edges.
    """
    width, height = image_size
    ordered = graph_features.sort_values(
        ["weighted_out_degree", "weighted_in_degree"], ascending=False
    )
    addresses = ordered.index.astype(str)
    rank = pd.Series(np.arange(len(addresses), dtype=np.int64), index=addresses)
    source_bins = np.asarray(rank // max(1, len(rank) // width), dtype=np.int64)
    target_bins = np.asarray(rank // max(1, len(rank) // height), dtype=np.int64)
    source_bins = np.minimum(source_bins, width - 1)
    target_bins = np.minimum(target_bins, height - 1)
    address_to_position = dict(zip(addresses, np.arange(len(addresses))))

    density = np.zeros((height, width), dtype=np.uint32)
    edge_count = 0
    missing_count = 0
    for chunk in pd.read_csv(
        edge_path,
        usecols=["input_address", "output_address"],
        dtype=str,
        chunksize=chunksize,
    ):
        source_ids = chunk["input_address"].map(address_to_position)
        target_ids = chunk["output_address"].map(address_to_position)
        valid = source_ids.notna() & target_ids.notna()
        missing_count += int((~valid).sum())
        source = source_bins[source_ids[valid].to_numpy(dtype=np.int64)]
        target = target_bins[target_ids[valid].to_numpy(dtype=np.int64)]
        np.add.at(density, (target, source), 1)
        edge_count += int(valid.sum())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = np.log1p(density.astype(np.float32))
    fig, ax = plt.subplots(figsize=(10, 9), dpi=180)
    plotted = ax.imshow(image, origin="lower", aspect="auto", cmap="magma")
    ax.set_title("Full Elliptic++ actor graph: degree-ordered edge density")
    ax.set_xlabel("Source actor rank (ordered by weighted out-degree)")
    ax.set_ylabel("Target actor rank (ordered by weighted out-degree)")
    fig.colorbar(plotted, ax=ax, label="log(1 + number of edges)")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return {
        "edges_rendered": edge_count,
        "edges_missing_actor": missing_count,
        "image_width": width,
        "image_height": height,
    }