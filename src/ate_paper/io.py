"""Small standard-library file helpers shared by preparation and analysis."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict], *, empty: str = "write") -> None:
    """Write a fresh CSV; retain each caller's policy for an empty population."""
    if empty not in {"write", "skip", "raise"}:
        raise ValueError("Unknown empty-CSV policy: " + empty)
    if not rows:
        if empty == "skip":
            return
        if empty == "raise":
            raise ValueError(f"Empty result: {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def file_receipt(path: Path, *, root: Path | None = None) -> dict:
    """Identify file bytes, using a relative name when a root is supplied."""
    name = str(path.relative_to(root)) if root is not None and path.is_relative_to(root) else str(path)
    return {"path": name, "bytes": path.stat().st_size, "sha256": sha256(path)}
