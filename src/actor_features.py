"""Build one leakage-free feature vector per actor from Elliptic++ snapshots."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


LABEL_COLUMN = "class"
ID_COLUMN = "address"
TIME_COLUMN = "Time step"


def load_actor_labels(path: Path) -> pd.Series:
    """Load actor labels indexed by address for evaluation only."""
    labels = pd.read_csv(path, usecols=[ID_COLUMN, LABEL_COLUMN])
    labels = labels.drop_duplicates(ID_COLUMN).set_index(ID_COLUMN)[LABEL_COLUMN]
    labels.name = LABEL_COLUMN
    return labels


def build_actor_features(path: Path) -> pd.DataFrame:
    """Aggregate temporal actor rows without using class labels.

    The resulting index is an address and every output column is numeric. The
    aggregation preserves both typical behaviour (mean) and activity scale
    (max), while keeping the feature matrix compact enough for clustering.
    """
    frame = pd.read_csv(path)
    feature_columns = [
        column
        for column in frame.columns
        if column not in {ID_COLUMN, LABEL_COLUMN, TIME_COLUMN}
    ]
    numeric = frame[feature_columns].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.replace([float("inf"), float("-inf")], pd.NA)
    frame = pd.concat([frame[[ID_COLUMN]], numeric], axis=1)

    means = frame.groupby(ID_COLUMN, sort=False)[feature_columns].mean()
    maxima = frame.groupby(ID_COLUMN, sort=False)[feature_columns].max()
    means.columns = [f"{column}__mean" for column in means.columns]
    maxima.columns = [f"{column}__max" for column in maxima.columns]
    result = pd.concat([means, maxima], axis=1)
    return result.apply(pd.to_numeric, errors="coerce")


def make_actor_dataset(features_path: Path, labels_path: Path) -> pd.DataFrame:
    """Return features with labels attached as a separate evaluation column."""
    features = build_actor_features(features_path)
    labels = load_actor_labels(labels_path)
    result = features.join(labels, how="left")
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    make_actor_dataset(args.features, args.labels).to_parquet(args.output)