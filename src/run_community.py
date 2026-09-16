"""Run and save the full actor-graph Leiden experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.actor_features import load_actor_labels
from src.community import fit_leiden
from src.configuration import load_experiment_config
from src.evaluation import cluster_enrichment
from src.manifest import write_manifest


def run_leiden(
    data_dir: Path,
    output_dir: Path,
    resolution: float = 1.0,
    seed: int = 42,
    config_path: Path = Path("config/experiment.json"),
) -> pd.DataFrame:
    """Fit Leiden and save memberships, summary, and enrichment tables."""
    actors = data_dir / "Actors_Dataset"
    addresses = pd.Index(
        pd.read_csv(actors / "wallets_classes.csv", usecols=["address"])["address"].astype(str),
        name="address",
    )
    memberships, summary = fit_leiden(
        actors / "AddrAddr_edgelist.csv", addresses, resolution=resolution, seed=seed
    )
    labels = load_actor_labels(actors / "wallets_classes.csv").reindex(memberships.index)
    enrichment = cluster_enrichment(memberships.to_numpy(), labels)
    output_dir.mkdir(parents=True, exist_ok=True)
    memberships.to_csv(output_dir / "leiden_memberships.csv", header=True)
    enrichment.to_csv(output_dir / "leiden_enrichment.csv")
    summary["resolution"] = resolution
    summary["seed"] = seed
    (output_dir / "leiden_summary.txt").write_text(
        "\n".join(f"{key}={value}" for key, value in summary.items()), encoding="utf-8"
    )
    write_manifest(
        output_dir,
        config_path,
        [
            actors / "wallets_classes.csv",
            actors / "AddrAddr_edgelist.csv",
        ],
        {
            "dataset": "Elliptic++",
            "resolution": resolution,
            "seed": seed,
            "method": "weighted_undirected_leiden",
        },
    )
    return enrichment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--config", type=Path, default=Path("config/experiment.json"))
    parser.add_argument("--resolution", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    config = load_experiment_config(args.config)
    leiden_config = config["leiden"]
    resolution = (
        args.resolution
        if args.resolution is not None
        else float(leiden_config["headline_resolution"])
    )
    seed = args.seed if args.seed is not None else int(config["random_state"])
    enrichment = run_leiden(args.data_dir, args.output_dir, resolution, seed, args.config)
    print(enrichment.head(20).to_string())


if __name__ == "__main__":
    main()