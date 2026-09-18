from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


ACTOR_CLASS_VALUES = frozenset({1, 2, 3})


@dataclass(frozen=True)
class TableValidation:
    name: str
    rows: int
    columns: int
    missing_cells: int
    duplicate_rows: int


def validate_table(df: pd.DataFrame, name: str) -> TableValidation:
    """Return non-destructive structural checks for a loaded table."""
    return TableValidation(
        name=name,
        rows=len(df),
        columns=len(df.columns),
        missing_cells=int(df.isna().sum().sum()),
        duplicate_rows=int(df.duplicated().sum()),
    )


def require_columns(df: pd.DataFrame, required: Iterable[str], name: str) -> None:
    missing = sorted(set(required).difference(df.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def validate_actor_tables(
    features: pd.DataFrame,
    classes: pd.DataFrame,
    addr_addr: pd.DataFrame,
) -> dict[str, TableValidation]:
    """Validate the keys and schemas needed by address-level clustering."""
    require_columns(features, {"address", "Time step"}, "wallets_features")
    require_columns(classes, {"address", "class"}, "wallets_classes")
    require_columns(
        addr_addr,
        {"input_address", "output_address"},
        "AddrAddr_edgelist",
    )

    if not classes["address"].is_unique:
        raise ValueError("wallets_classes.address must be unique")

    invalid_classes = set(classes["class"].dropna().unique()).difference(
        ACTOR_CLASS_VALUES
    )
    if invalid_classes:
        raise ValueError(f"wallets_classes contains invalid class values: {sorted(invalid_classes)}")

    feature_addresses = set(features["address"].dropna())
    class_addresses = set(classes["address"].dropna())
    if feature_addresses != class_addresses:
        raise ValueError("wallets_features and wallets_classes address keys do not match")

    return {
        "wallets_features": validate_table(features, "wallets_features"),
        "wallets_classes": validate_table(classes, "wallets_classes"),
        "AddrAddr_edgelist": validate_table(addr_addr, "AddrAddr_edgelist"),
    }