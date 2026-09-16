"""Audit the Elliptic++ CSV files without loading edge lists into memory."""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CsvSummary:
    path: str
    rows: int
    columns: list[str]
    sample: list[dict[str, str]]


def summarize_csv(path: Path, sample_size: int = 3) -> CsvSummary:
    """Return schema and a small sample for a CSV file."""
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        sample: list[dict[str, str]] = []
        rows = 0
        for row in reader:
            rows += 1
            if len(sample) < sample_size:
                sample.append(row)
        return CsvSummary(str(path), rows, columns, sample)


def value_counts(path: Path, column: str) -> Counter[str]:
    """Count values in one CSV column using a single streaming pass."""
    counts: Counter[str] = Counter()
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle)
        if column not in (reader.fieldnames or []):
            raise ValueError(f"Missing column {column!r} in {path}")
        for row in reader:
            counts[row[column]] += 1
    return counts


def edge_summary(path: Path, source_column: str, target_column: str) -> dict[str, int]:
    """Summarize an edge list, including duplicate edges and self-loops."""
    edges = 0
    self_loops = 0
    unique_edges: set[tuple[str, str]] = set()
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle)
        expected = {source_column, target_column}
        if not expected.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing edge columns {expected} in {path}")
        for row in reader:
            source = row[source_column]
            target = row[target_column]
            edges += 1
            self_loops += source == target
            unique_edges.add((source, target))
    return {
        "edges": edges,
        "unique_edges": len(unique_edges),
        "duplicate_edges": edges - len(unique_edges),
        "self_loops": self_loops,
    }


def audit_dataset(data_dir: Path) -> dict[str, object]:
    """Collect the checks needed before building actor-level representations."""
    actors = data_dir / "Actors_Dataset"
    transactions = data_dir / "Transactions_Dataset"
    actor_classes = actors / "wallets_classes.csv"
    transaction_classes = transactions / "txs_classes.csv"
    combined_features = actors / "wallets_features_classes_combined.csv"
    actor_edges = actors / "AddrAddr_edgelist.csv"

    return {
        "files": {
            "actor_features": summarize_csv(combined_features).__dict__,
            "actor_classes": summarize_csv(actor_classes).__dict__,
            "transaction_classes": summarize_csv(transaction_classes).__dict__,
            "actor_edges": summarize_csv(actor_edges).__dict__,
        },
        "actor_class_counts": dict(value_counts(actor_classes, "class")),
        "transaction_class_counts": dict(value_counts(transaction_classes, "class")),
        "actor_edge_summary": edge_summary(
            actor_edges, "input_address", "output_address"
        ),
    }


def main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing Actors_Dataset and Transactions_Dataset.",
    )
    args = parser.parse_args()
    print(json.dumps(audit_dataset(args.data_dir), indent=2))


if __name__ == "__main__":
    main()