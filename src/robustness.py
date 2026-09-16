"""Robustness and null-model tests for illicit community enrichment."""

from __future__ import annotations

import numpy as np
import pandas as pd


def observed_enrichment_by_size(
    memberships: pd.Series | np.ndarray,
    labels: pd.Series | np.ndarray,
    minimum_size: int = 30,
) -> pd.DataFrame:
    """Return community enrichment after applying a minimum known-label size."""
    frame = pd.DataFrame(
        {"community": np.asarray(memberships), "label": np.asarray(labels)}
    )
    frame = frame[frame["label"].isin([1, 2])]
    grouped = frame.groupby("community")
    result = grouped["label"].agg(
        known_count="size",
        illicit_count=lambda values: int((values == 1).sum()),
    )
    result["illicit_rate"] = result["illicit_count"] / result["known_count"]
    result["global_illicit_rate"] = float((frame["label"] == 1).mean())
    result["enrichment"] = result["illicit_rate"] / result["global_illicit_rate"]
    result = result[result["known_count"] >= minimum_size]
    return result.sort_values("enrichment", ascending=False)


def permutation_enrichment_test(
    memberships: pd.Series | np.ndarray,
    labels: pd.Series | np.ndarray,
    minimum_size: int = 30,
    permutations: int = 100,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Test whether the strongest enrichment exceeds label-shuffling nulls.

    Only known labels are shuffled, preserving the unknown-label mask and the
    exact illicit/licit class counts. Community assignments remain untouched.
    """
    membership_values = np.asarray(memberships)
    label_values = np.asarray(labels)
    known_mask = np.isin(label_values, [1, 2])
    communities, inverse = np.unique(membership_values, return_inverse=True)
    known_counts = np.bincount(inverse[known_mask], minlength=len(communities))
    eligible = known_counts >= minimum_size
    if not np.any(eligible):
        raise ValueError(
            "No community has enough known labels for the requested minimum_size."
        )
    known_illicit = label_values[known_mask] == 1
    global_rate = float(known_illicit.mean())
    rng = np.random.default_rng(random_state)
    null_maxima = np.empty(permutations, dtype=np.float64)
    for iteration in range(permutations):
        shuffled = rng.permutation(known_illicit)
        illicit_counts = np.bincount(
            inverse[known_mask], weights=shuffled.astype(np.int64), minlength=len(communities)
        )
        rates = illicit_counts / np.maximum(known_counts, 1)
        null_maxima[iteration] = float(np.max(rates[eligible] / global_rate))
    observed = observed_enrichment_by_size(membership_values, label_values, minimum_size)
    observed_max = float(observed["enrichment"].max()) if len(observed) else np.nan
    p_value = float((1 + np.sum(null_maxima >= observed_max)) / (permutations + 1))
    summary = {
        "minimum_size": minimum_size,
        "permutations": permutations,
        "observed_max_enrichment": observed_max,
        "null_mean_max_enrichment": float(null_maxima.mean()),
        "null_95_max_enrichment": float(np.quantile(null_maxima, 0.95)),
        "empirical_p_value": p_value,
    }
    null_frame = pd.DataFrame({"null_max_enrichment": null_maxima})
    return null_frame, summary