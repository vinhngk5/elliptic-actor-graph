"""Compute scalable actor-graph statistics from the Elliptic++ edge list."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd


def build_graph_features(
    path: Path,
    chunksize: int = 250_000,
) -> pd.DataFrame:
    """Return weighted and directed degree features indexed by address.

    Each input row is treated as one observed relation. Duplicate rows are
    therefore retained in weighted degree, while unique-neighbour counts are
    calculated from sets and self-loops are excluded from neighbour counts.
    """
    weighted_out: Counter[str] = Counter()
    weighted_in: Counter[str] = Counter()
    out_neighbours: dict[str, set[str]] = {}
    in_neighbours: dict[str, set[str]] = {}
    self_loops: Counter[str] = Counter()

    for chunk in pd.read_csv(
        path,
        usecols=["input_address", "output_address"],
        dtype=str,
        chunksize=chunksize,
    ):
        sources = chunk["input_address"].tolist()
        targets = chunk["output_address"].tolist()
        for source, target in zip(sources, targets):
            weighted_out[source] += 1
            weighted_in[target] += 1
            if source == target:
                self_loops[source] += 1
                continue
            out_neighbours.setdefault(source, set()).add(target)
            in_neighbours.setdefault(target, set()).add(source)

    addresses = set(weighted_out) | set(weighted_in)
    result = pd.DataFrame(index=pd.Index(sorted(addresses), name="address"))
    result["weighted_out_degree"] = pd.Series(weighted_out, dtype="int64")
    result["weighted_in_degree"] = pd.Series(weighted_in, dtype="int64")
    result["out_degree"] = pd.Series(
        {address: len(neighbours) for address, neighbours in out_neighbours.items()},
        dtype="int64",
    )
    result["in_degree"] = pd.Series(
        {address: len(neighbours) for address, neighbours in in_neighbours.items()},
        dtype="int64",
    )
    result["self_loop_count"] = pd.Series(self_loops, dtype="int64")
    return result.fillna(0).astype("int64")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edges", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build_graph_features(args.edges).to_parquet(args.output)