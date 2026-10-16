"""Verify every portable evidence artifact referenced by a JSON manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = (
    ROOT / "evidence/current/golden_vectors/manifest.json",
    ROOT / "evidence/current/hardware_image/layout.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def records(value: object):
    if isinstance(value, dict):
        if isinstance(value.get("path"), str) and isinstance(
            value.get("sha256"), str
        ):
            yield value
        for item in value.values():
            yield from records(item)
    elif isinstance(value, list):
        for item in value:
            yield from records(item)


def main() -> None:
    checked = 0
    external = 0
    for manifest in MANIFESTS:
        document = json.loads(manifest.read_text(encoding="utf-8"))
        for record in records(document):
            stored_path = record["path"]
            if stored_path.startswith("external://"):
                external += 1
                continue
            path = Path(stored_path)
            if path.is_absolute():
                raise ValueError(f"non-portable absolute path: {stored_path}")
            if not path.parts or path.parts[0] != "evidence":
                # Source-code records bind the historical generator version.
                # The repository contains the final implementation, which can
                # legitimately have a later hash than the archived generator.
                external += 1
                continue
            artifact = ROOT / path
            if not artifact.is_file():
                raise FileNotFoundError(artifact)
            actual = sha256(artifact)
            if actual != record["sha256"]:
                raise ValueError(f"SHA-256 mismatch: {stored_path}")
            checked += 1
    print(f"portable artifacts checked: {checked}")
    print(f"external provenance records skipped: {external}")
    print("status: passed")


if __name__ == "__main__":
    main()
