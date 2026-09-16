import numpy as np
import pandas as pd

from src.evaluation import cluster_enrichment
from src.graph_features import build_graph_features


def test_cluster_enrichment_reports_unknown_membership() -> None:
    result = cluster_enrichment(np.array([0, 0, 0, 1]), pd.Series([1, 2, 3, 3]))

    assert result.loc[0, "total_count"] == 3
    assert result.loc[0, "known_count"] == 2
    assert result.loc[0, "unknown_count"] == 1
    assert result.loc[0, "illicit_count"] == 1


def test_graph_features_keep_weight_and_exclude_self_loop_neighbours(tmp_path) -> None:
    path = tmp_path / "edges.csv"
    pd.DataFrame(
        {
            "input_address": ["a", "a", "a", "b"],
            "output_address": ["b", "b", "a", "a"],
        }
    ).to_csv(path, index=False)

    result = build_graph_features(path, chunksize=2)

    assert result.loc["a", "weighted_out_degree"] == 3
    assert result.loc["a", "out_degree"] == 1
    assert result.loc["a", "self_loop_count"] == 1
    assert result.loc["b", "in_degree"] == 1


def test_configuration_excludes_class_from_numeric_matrix() -> None:
    from src.clustering import prepare_matrix

    frame = pd.DataFrame(
        {
            "behaviour": [1.0, 2.0, 3.0],
            "class": [1, 2, 3],
            "address_text": ["a", "b", "c"],
        }
    )

    matrix, _, _, _ = prepare_matrix(frame, n_components=1)

    assert matrix.shape == (3, 1)
