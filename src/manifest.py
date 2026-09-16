"""Write provenance manifests for reproducible experiment runs."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of a file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision() -> str | None:
    """Return the current git revision when the run is inside a repository."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_manifest(
    output_dir: Path,
    config_path: Path,
    input_files: Iterable[Path],
    parameters: dict[str, Any],
) -> None:
    """Persist configuration, code, environment, and input-file provenance."""
    manifest = {
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "git_revision": git_revision(),
        "python": sys.version,
        "platform": platform.platform(),
        "parameters": parameters,
        "input_files": {
            str(path): {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in input_files
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )