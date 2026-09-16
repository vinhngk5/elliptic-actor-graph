"""Temporal stability checks for actor representations and clusters."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import adjusted_rand_score

from src.clustering import fit_kmeans, prepare_matrix


def build_window_features(
    path: Path,
    addresses: pd.Index,
    minimum_timestep: int | None = None,
    maximum_timestep: int | None = None,
) -> pd.DataFrame:
    """Aggregate only selected actors and a timestep window."""
    selected = set(addresses.astype(str))
    frame = pd.read_csv(path)
    frame["Time step"] = pd.to_numeric(frame["Time step"], errors="coerce")
    frame = frame[frame["address"].astype(str).isin(selected)]
    if minimum_timestep is not None:
        frame = frame[frame["Time step"] >= minimum_timestep]
    if maximum_timestep is not None:
        frame = frame[frame["Time step"] <= maximum_timestep]
    if frame.empty:
        raise ValueError("The requested timestep window contains no actor records.")
    feature_columns = [
        column for column in frame.columns if column not in {"address", "class", "Time step"}
    ]
    numeric = frame[feature_columns].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.replace([float("inf"), float("-inf")], pd.NA)
    numeric.index = frame["address"].to_numpy()
    means = numeric.groupby(level=0, sort=False).mean()
    maxima = numeric.groupby(level=0, sort=False).max()
    means.columns = [f"{column}__mean" for column in means.columns]
    maxima.columns = [f"{column}__max" for column in maxima.columns]
    return pd.concat([means, maxima], axis=1)


def temporal_stability(
    features_path: Path,
    labels: pd.Series,
    addresses: pd.Index,
    split_timestep: int,
    n_clusters: int = 8,
    random_state: int = 42,
) -> dict[str, float | int]:
    """Compare independently fitted clusters before and after a time split."""
    early = build_window_features(features_path, addresses, maximum_timestep=split_timestep)
    late = build_window_features(features_path, addresses, minimum_timestep=split_timestep + 1)
    common = early.index.intersection(late.index).intersection(labels.index)
    if len(common) < max(100, n_clusters * 10):
        raise ValueError(f"Only {len(common)} actors overlap between temporal windows.")
    difference = (early.loc[common] - late.loc[common]).abs().to_numpy()
    if float(difference.mean()) < 1e-12:
        raise ValueError(
            "Early and late feature windows are numerically identical; "
            "temporal stability would be invalid."
        )
    early_matrix, _, _, _ = prepare_matrix(early.loc[common], random_state=random_state)
    late_matrix, _, _, _ = prepare_matrix(late.loc[common], random_state=random_state)
    early_clusters = fit_kmeans(early_matrix, n_clusters, random_state)
    late_clusters = fit_kmeans(late_matrix, n_clusters, random_state)
    known = labels.loc[common].to_numpy() != 3
    return {
        "split_timestep": split_timestep,
        "common_actors": int(len(common)),
        "cluster_ari_between_windows": float(
            adjusted_rand_score(early_clusters, late_clusters)
        ),
        "early_known_actors": int(known.sum()),
    }