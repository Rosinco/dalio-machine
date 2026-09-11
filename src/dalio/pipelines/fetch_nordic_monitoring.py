"""Capture twelve original Nordic monitoring inputs without opening the database."""

import argparse
import json
from pathlib import Path

from dalio.nordic_monitoring.acquisition import collect_bundle, load_bundle


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=Path("data/artifacts/nordic_monitoring"))
    parser.add_argument("--pacing-seconds", type=float, default=1.0)
    args = parser.parse_args(argv)
    path = collect_bundle(artifact_root=args.artifact_root, pacing_seconds=args.pacing_seconds)
    result = load_bundle(path)
    print(json.dumps({k: v for k, v in result.items() if k not in {"series", "protected_artifact_paths"}}
                     | {"signals_requested": 12, "signals_successful": len(result["series"])}, indent=2))
    return int(bool(result["gaps"]))


if __name__ == "__main__":
    raise SystemExit(main())
