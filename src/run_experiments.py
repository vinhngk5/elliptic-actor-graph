"""Run the resource-bounded actor clustering benchmark."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import hdbscan
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.actor_features import make_actor_dataset
from src.clustering import fit_kmeans, prepare_matrix
from src.configuration import load_experiment_config
from src.evaluation import cluster_enrichment, clustering_metrics
from src.graph_features import build_graph_features
from src.manifest import write_manifest


def run_benchmark(
    data_dir: Path,
    output_dir: Path,
    sample_size: int = 20_000,
    random_state: int = 42,
    kmeans_clusters: tuple[int, ...] = (4, 8, 12),
    hdbscan_min_cluster_size: int = 50,
    hdbscan_min_samples: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run KMeans and HDBSCAN on one fixed actor sample."""
    actors = data_dir / "Actors_Dataset"
    actor = make_actor_dataset(
        actors / "wallets_features_classes_combined.csv",
        actors / "wallets_classes.csv",
    )
    graph = build_graph_features(actors / "AddrAddr_edgelist.csv")
    actor = actor.join(graph, how="left").fillna(0)
    sample = actor.sample(n=min(sample_size, len(actor)), random_state=random_state)
    labels = sample["class"].copy()
    matrix, pca, _, _ = prepare_matrix(sample, random_state=random_state)

    assignments: dict[str, object] = {}
    for n_clusters in kmeans_clusters:
        assignments[f"kmeans_{n_clusters}"] = fit_kmeans(
            matrix, n_clusters=n_clusters, random_state=random_state
        )
    assignments["hdbscan"] = hdbscan.HDBSCAN(
        min_cluster_size=hdbscan_min_cluster_size,
        min_samples=hdbscan_min_samples,
        cluster_selection_method="eom",
        core_dist_n_jobs=2,
    ).fit_predict(matrix)

    metric_rows = []
    enrichment_frames = []
    for method, cluster_ids in assignments.items():
        metrics = clustering_metrics(matrix, cluster_ids, labels, random_state=random_state)
        metrics["method"] = method
        metrics["sample_size"] = len(sample)
        metrics["pca_variance"] = float(pca.explained_variance_ratio_.sum())
        metric_rows.append(metrics)
        enrichment = cluster_enrichment(cluster_ids, labels).reset_index()
        enrichment["method"] = method
        enrichment_frames.append(enrichment)

    metrics_frame = pd.DataFrame(metric_rows).set_index("method")
    enrichment_frame = pd.concat(enrichment_frames, ignore_index=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_frame.to_csv(output_dir / "clustering_metrics.csv")
    enrichment_frame.to_csv(output_dir / "cluster_enrichment.csv", index=False)

    sns.set_theme(style="whitegrid", context="notebook")
    plot_frame = metrics_frame.reset_index().melt(
        id_vars="method", value_vars=["silhouette", "ari_known", "nmi_known"]
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=plot_frame, x="method", y="value", hue="variable", ax=ax)
    ax.set_title("Actor clustering benchmark")
    ax.set_ylabel("Score")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(output_dir / "clustering_benchmark.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    del actor, graph, sample, matrix
    gc.collect()
    return metrics_frame, enrichment_frame


def write_experiment_manifest(
    output_dir: Path,
    sample_size: int,
    random_state: int,
    config_path: Path,
    kmeans_clusters: tuple[int, ...],
    hdbscan_min_cluster_size: int,
    hdbscan_min_samples: int,
) -> None:
    """Record the fixed settings needed to reproduce the benchmark."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "experiment_manifest.txt").write_text(
        "dataset=Elliptic++\n"
        f"sample_size={sample_size}\n"
        f"random_state={random_state}\n"
        f"config={config_path}\n"
        f"methods={','.join(f'kmeans_{value}' for value in kmeans_clusters)},hdbscan\n"
        f"hdbscan_min_cluster_size={hdbscan_min_cluster_size}\n"
        f"hdbscan_min_samples={hdbscan_min_samples}\n"
        "unknown_label=3\n"
        "feature_labels_excluded=true\n"
        "graph_features=weighted_out_degree,weighted_in_degree,out_degree,in_degree,self_loop_count\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--sample-size", type=int, default=20_000)
    parser.add_argument("--config", type=Path, default=Path("config/experiment.json"))
    args = parser.parse_args()
    config = load_experiment_config(args.config)
    clustering_config = config["clustering"]
    random_state = int(config["random_state"])
    sample_size = args.sample_size if args.sample_size != 20_000 else int(config["sample_size"])
    kmeans_clusters = tuple(int(value) for value in clustering_config["kmeans_clusters"])
    hdbscan_min_cluster_size = int(clustering_config["hdbscan_min_cluster_size"])
    hdbscan_min_samples = int(clustering_config["hdbscan_min_samples"])
    metrics, enrichment = run_benchmark(
        args.data_dir,
        args.output_dir,
        sample_size=sample_size,
        random_state=random_state,
        kmeans_clusters=kmeans_clusters,
        hdbscan_min_cluster_size=hdbscan_min_cluster_size,
        hdbscan_min_samples=hdbscan_min_samples,
    )
    write_experiment_manifest(
        args.output_dir,
        sample_size,
        random_state,
        args.config,
        kmeans_clusters,
        hdbscan_min_cluster_size,
        hdbscan_min_samples,
    )
    write_manifest(
        args.output_dir,
        args.config,
        [
            args.config,
            args.data_dir / "Actors_Dataset" / "wallets_features_classes_combined.csv",
            args.data_dir / "Actors_Dataset" / "wallets_classes.csv",
            args.data_dir / "Actors_Dataset" / "AddrAddr_edgelist.csv",
        ],
        {
            "dataset": config["dataset"],
            "sample_size": sample_size,
            "random_state": random_state,
            "kmeans_clusters": kmeans_clusters,
            "hdbscan_min_cluster_size": hdbscan_min_cluster_size,
            "hdbscan_min_samples": hdbscan_min_samples,
        },
    )
    print(metrics.to_string())
    print("\nTop enrichment clusters:")
    print(enrichment.head(20).to_string(index=False))


if __name__ == "__main__":
    main()