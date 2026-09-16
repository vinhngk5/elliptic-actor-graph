"""Evaluation metrics for actor clustering experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)


def clustering_metrics(
    matrix: np.ndarray,
    assignments: np.ndarray,
    labels: pd.Series,
    sample_size: int = 10_000,
    random_state: int = 42,
) -> dict[str, float | int]:
    """Compute geometry metrics and known-label agreement metrics."""
    assignments = np.asarray(assignments)
    valid = assignments >= 0
    known = labels.to_numpy() != 3
    evaluate = valid & known
    unique_clusters = np.unique(assignments[valid])
    result: dict[str, float | int] = {
        "n_clusters": int(len(unique_clusters)),
        "noise_fraction": float(np.mean(~valid)),
    }
    if len(unique_clusters) > 1 and valid.sum() > 2:
        metric_matrix = matrix[valid]
        metric_assignments = assignments[valid]
        result["silhouette"] = float(
            silhouette_score(
                metric_matrix,
                metric_assignments,
                sample_size=min(sample_size, len(metric_matrix)),
                random_state=random_state,
            )
        )
        result["davies_bouldin"] = float(
            davies_bouldin_score(metric_matrix, metric_assignments)
        )
        result["calinski_harabasz"] = float(
            calinski_harabasz_score(metric_matrix, metric_assignments)
        )
    else:
        result.update(
            {"silhouette": np.nan, "davies_bouldin": np.nan, "calinski_harabasz": np.nan}
        )
    if evaluate.sum() > 0:
        result["ari_known"] = float(
            adjusted_rand_score(labels.to_numpy()[evaluate], assignments[evaluate])
        )
        result["nmi_known"] = float(
            normalized_mutual_info_score(labels.to_numpy()[evaluate], assignments[evaluate])
        )
    else:
        result.update({"ari_known": np.nan, "nmi_known": np.nan})
    return result


def cluster_enrichment(assignments: np.ndarray, labels: pd.Series) -> pd.DataFrame:
    """Summarize illicit rate and enrichment for each non-noise cluster.

    Counts are reported separately for all assigned actors and actors with a
    known evaluation label. Unknown labels must not disappear from the cluster
    denominator without being visible in the output.
    """
    frame = pd.DataFrame({"cluster": assignments, "label": labels.to_numpy()})
    frame = frame[frame["cluster"] >= 0]
    known = frame[frame["label"].isin([1, 2])]
    global_rate = float((known["label"] == 1).mean()) if len(known) else np.nan
    total_counts = frame.groupby("cluster").size().rename("total_count")
    known_counts = known.groupby("cluster").size().rename("known_count")
    illicit_counts = (
        known.assign(illicit=known["label"] == 1)
        .groupby("cluster")["illicit"]
        .sum()
        .rename("illicit_count")
    )
    result = pd.concat([total_counts, known_counts, illicit_counts], axis=1).fillna(0)
    result[["total_count", "known_count", "illicit_count"]] = result[
        ["total_count", "known_count", "illicit_count"]
    ].astype(int)
    result["unknown_count"] = result["total_count"] - result["known_count"]
    result["illicit_rate"] = result["illicit_count"] / result["known_count"]
    result["global_illicit_rate"] = global_rate
    result["enrichment"] = result["illicit_rate"] / global_rate
    return result.sort_values("enrichment", ascending=False)