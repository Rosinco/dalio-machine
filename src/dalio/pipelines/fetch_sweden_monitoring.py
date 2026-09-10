"""Capture a complete immutable Sweden monitoring supplement, without DB writes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dalio.monitoring.acquisition import INDICATORS, collect_bundle, load_bundle


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True,
                        help="Directory for immutable raw responses and complete capture bundles")
    args = parser.parse_args(argv)
    path = collect_bundle(artifact_root=args.artifact_root)
    checked = load_bundle(path)
    print(json.dumps({"bundle_path": checked["bundle_path"], "bundle_sha256": checked["bundle_sha256"],
        "available_at": checked["available_at"], "requested_series": len(INDICATORS),
        "successful_series": len(checked["series"]), "gaps": checked["gaps"]}, indent=2, allow_nan=False))
    return 1 if checked["gaps"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
