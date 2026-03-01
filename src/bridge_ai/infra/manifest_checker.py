"""Small entrypoint for manifest consistency checks."""

from __future__ import annotations

import argparse
from pathlib import Path
import yaml
from bridge_ai.data.manifest import validate_manifest


def run(manifest_path: str = "artifacts/manifest.json"):
    return validate_manifest(manifest_path)


def _parse_args() -> argparse.Namespace:  # pragma: no cover
    parser = argparse.ArgumentParser(description="Check manifest reproducibility metadata.")
    parser.add_argument("--manifest-path", "--manifest_path", default="artifacts/manifest.json")
    parser.add_argument("--config-path", "--config_path", default=None)
    return parser.parse_args()


def main() -> None:  # pragma: no cover
    args = _parse_args()
    manifest_path = args.manifest_path
    if args.config_path:
        cfg = yaml.safe_load(Path(args.config_path).read_text(encoding="utf-8"))
        manifest_path = cfg.get("storage", {}).get("manifest_path", args.manifest_path)
    issues = run(manifest_path)
    if issues:
        print({"manifest_issues": issues})
    else:
        print({"manifest_issues": []})


if __name__ == "__main__":
    main()
