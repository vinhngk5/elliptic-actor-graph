"""Community-conditioned multi-view actor discovery utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


def community_conditioned_residuals(
    embedding: np.ndarray,
    communities: pd.Series | np.ndarray,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Measure behavioural deviation from each actor's graph community.

    This is unsupervised: labels are not used. For each community, a centroid
    is estimated in the behavioural embedding, then every actor receives its
    Euclidean distance to that centroid. The distance is a cross-view signal:
    graph membership supplies context and behaviour supplies the deviation.
    """
    values = np.asarray(embedding, dtype=np.float32)
    community_ids = np.asarray(communities)
    if len(values) != len(community_ids):
        raise ValueError("Embedding rows and community assignments must have equal length.")
    unique, inverse = np.unique(community_ids, return_inverse=True)
    centroids = np.zeros((len(unique), values.shape[1]), dtype=np.float32)
    counts = np.bincount(inverse).astype(np.int64)
    np.add.at(centroids, inverse, values)
    centroids /= counts[:, None]
    residuals = np.linalg.norm(values - centroids[inverse], axis=1)
    stats = pd.DataFrame(
        {
            "community": unique,
            "community_size": counts,
            "mean_behavioral_residual": np.bincount(
                inverse, weights=residuals
            ) / counts,
        }
    ).sort_values("mean_behavioral_residual", ascending=False)
    return residuals, stats


def robust_zscore(values: np.ndarray) -> np.ndarray:
    """Scale a score with median and MAD, robust to heavy-tailed graph data."""
    values = np.asarray(values, dtype=np.float32)
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    return (values - median) / max(1.4826 * mad, 1e-6)