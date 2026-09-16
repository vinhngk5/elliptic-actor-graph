"""Reproducible preprocessing and baseline clustering for actor vectors."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler


def prepare_matrix(
    frame: pd.DataFrame,
    label_column: str = "class",
    n_components: int = 30,
    random_state: int = 42,
) -> tuple[np.ndarray, PCA, RobustScaler, SimpleImputer]:
    """Transform actor features into a robust, low-dimensional matrix."""
    features = frame.drop(columns=[label_column], errors="ignore")
    features = features.select_dtypes(include=["number"])
    features = features.loc[:, features.nunique(dropna=False) > 1]

    imputer = SimpleImputer(strategy="median")
    values = imputer.fit_transform(features)
    values = np.sign(values) * np.log1p(np.abs(values))

    scaler = RobustScaler()
    scaled = scaler.fit_transform(values)
    component_count = min(n_components, scaled.shape[0], scaled.shape[1])
    pca = PCA(n_components=component_count, random_state=random_state)
    return pca.fit_transform(scaled), pca, scaler, imputer


def fit_kmeans(
    matrix: np.ndarray,
    n_clusters: int,
    random_state: int = 42,
) -> np.ndarray:
    """Fit the paper's first clustering baseline."""
    model = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=10,
        random_state=random_state,
    )
    return model.fit_predict(matrix)