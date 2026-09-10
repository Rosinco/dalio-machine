"""Package saved Dalio outputs for offline import; never fetch or open a database."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from business_model import business_index
from export_snapshot import export

from dalio.app.fundamentals.snapshot import parse_snapshot


def document(path: Path) -> dict:
    contents = path.read_bytes()
    return {
        "source_file": path.name,
        "sha256": hashlib.sha256(contents).hexdigest(),
        "content": contents.decode("utf-8"),
    }


def package(
    fundamentals: Path, liquidity: Path | None = None, business: Path | None = None
) -> tuple[dict, dict]:
    fund = document(fundamentals)
    raw = json.loads(fund["content"])
    parse_snapshot(raw)
    liquid = document(liquidity) if liquidity else None
    liquid_raw = json.loads(liquid["content"]) if liquid else None
    if liquid_raw and (
        liquid_raw.get("version") != 1
        or liquid_raw.get("methodology_version") != "liquidity-diagnostics-v1"
    ):
        raise ValueError("Unsupported liquidity methodology")
    company = document(business) if business else None
    company_raw = json.loads(company["content"]) if company else None
    if company_raw and company_raw.get("version") != 1:
        raise ValueError("Unsupported business research format")
    version = 2 if company else 1
    identity = (
        f"macro-atlas-research-v{version}\n{fund['sha256']}\n{liquid['sha256'] if liquid else ''}"
    )
    if version == 2:
        identity += f"\n{company['sha256']}"
    release = {
        "id": hashlib.sha256(identity.encode()).hexdigest(),
        "as_of": raw["as_of"],
        "generated_at": raw["generated_at"],
        "fundamentals_sha256": fund["sha256"],
        "liquidity_as_of": liquid_raw["as_of"] if liquid_raw else None,
        "liquidity_sha256": liquid["sha256"] if liquid else None,
        "business_as_of": company_raw["as_of"] if company_raw else None,
        "business_sha256": company["sha256"] if company else None,
        "company_count": len(company_raw["companies"]) if company_raw else 0,
        "country_count": len(raw["countries"]),
        "indicator_count": len(raw["indicators"]),
    }
    payload = {
        "format": "macro-atlas-research",
        "schema_version": version,
        "fundamentals": fund,
        "liquidity": liquid,
    }
    if company:
        payload["business"] = company
    return payload, release


def encode(payload: dict) -> str:
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    if len(text.encode()) > 32 * 1024 * 1024:
        raise ValueError("Macro Atlas research packages must be smaller than 32 MB")
    return text


def bundle(
    fundamentals: Path,
    liquidity: Path | None,
    previous: list[Path],
    output: Path,
    business: Path | None = None,
) -> None:
    catalogue = {"version": 1, "default_id": "", "releases": []}
    for position, source in enumerate([fundamentals, *previous]):
        selected_liquidity = liquidity if position == 0 else None
        payload, release = package(source, selected_liquidity, business if position == 0 else None)
        destination = output if position == 0 else output / "releases" / release["id"]
        export(source, destination)
        (destination / "research.atlas.json").write_text(encode(payload), encoding="utf-8")
        if selected_liquidity:
            (destination / "liquidity.json").write_bytes(selected_liquidity.read_bytes())
        if payload.get("business"):
            raw = json.loads(payload["business"]["content"])
            (destination / "business-index.json").write_text(
                json.dumps(business_index(raw), ensure_ascii=False, allow_nan=False),
                encoding="utf-8",
            )
            (destination / "business-research.json").write_text(
                json.dumps(raw["research"], ensure_ascii=False, allow_nan=False), encoding="utf-8"
            )
            (destination / "companies").mkdir(exist_ok=True)
            for identifier, company in raw["companies"].items():
                if not identifier.isdigit() or len(identifier) > 10 or identifier.startswith("0"):
                    raise ValueError("Invalid company identifier")
                (destination / "companies" / f"{identifier}.json").write_text(
                    json.dumps(company, ensure_ascii=False, allow_nan=False), encoding="utf-8"
                )
        relative = "./data" if position == 0 else f"./data/releases/{release['id']}"
        catalogue["releases"].append(
            {
                **release,
                "storage": "included",
                "base": relative,
                "package_url": f"{relative}/research.atlas.json",
            }
        )
        if position == 0:
            catalogue["default_id"] = release["id"]
    (output / "catalog.json").write_text(json.dumps(catalogue, indent=2), encoding="utf-8")
    print(
        json.dumps({"default_id": catalogue["default_id"], "releases": len(catalogue["releases"])})
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fundamentals", type=Path, required=True)
    parser.add_argument("--liquidity", type=Path)
    parser.add_argument(
        "--business", type=Path, help="Selected company and branch research exported from Börsdata"
    )
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--output", type=Path, help="Write one importable .atlas.json file")
    destination.add_argument(
        "--bundle", type=Path, help="Build included releases under public/data"
    )
    parser.add_argument("--previous", type=Path, action="append", default=[])
    args = parser.parse_args()
    if args.bundle:
        bundle(args.fundamentals, args.liquidity, args.previous, args.bundle, args.business)
    else:
        payload, release = package(args.fundamentals, args.liquidity, args.business)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encode(payload), encoding="utf-8")
        print(json.dumps(release, indent=2))
