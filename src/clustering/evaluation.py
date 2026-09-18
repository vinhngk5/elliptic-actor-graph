from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
	adjusted_mutual_info_score,
	adjusted_rand_score,
	calinski_harabasz_score,
	davies_bouldin_score,
	normalized_mutual_info_score,
	silhouette_score,
)


def evaluate_clusters(
	assignments: pd.DataFrame,
	features: np.ndarray | None = None,
	labels: pd.Series | None = None,
) -> dict[str, float | int | None]:
	"""Compute representation-aware internal and optional external metrics."""
	cluster_labels = assignments["cluster_id"].to_numpy()
	result: dict[str, float | int | None] = {
		"n_clusters": int(np.unique(cluster_labels).size),
		"silhouette": None,
		"davies_bouldin": None,
		"calinski_harabasz": None,
		"ari": None,
		"nmi": None,
		"ami": None,
	}
	if features is not None and len(np.unique(cluster_labels)) > 1:
		result["silhouette"] = float(silhouette_score(features, cluster_labels))
		result["davies_bouldin"] = float(davies_bouldin_score(features, cluster_labels))
		result["calinski_harabasz"] = float(calinski_harabasz_score(features, cluster_labels))
	if labels is not None:
		known = labels.to_numpy() != 3
		if known.sum() > 0 and np.unique(cluster_labels[known]).size > 1:
			result["ari"] = float(adjusted_rand_score(labels.to_numpy()[known], cluster_labels[known]))
			result["nmi"] = float(normalized_mutual_info_score(labels.to_numpy()[known], cluster_labels[known]))
			result["ami"] = float(adjusted_mutual_info_score(labels.to_numpy()[known], cluster_labels[known]))
	return result
