"""Checked Version 1 denominator for sovereign refinancing evidence.

The denominator is deliberately separate from provider adapters.  It freezes
what counts as the bounded package while allowing the 31 machine-readable
Eurostat/ECB scalar partitions to land before the heterogeneous national debt
office streams.  Runtime storage status is not written into this file: an
unavailable official source remains a gap inside the fixed denominator.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from urllib.parse import urlsplit

SOVEREIGN_REFINANCING_MANIFEST_SCHEMA_VERSION = 1
SOVEREIGN_REFINANCING_MANIFEST_ID = "sovereign_refinancing_v1"
SOVEREIGN_REFINANCING_PARTITION_COUNT = 48
SOVEREIGN_REFINANCING_SCALAR_PARTITION_COUNT = 31
SOVEREIGN_REFINANCING_NATIVE_PARTITION_COUNT = 17
SOVEREIGN_REFINANCING_VINTAGE_PREFIX = "sovereign-refinancing-catalogue-sha256:"
SOVEREIGN_REFINANCING_MANIFEST_SHA256 = (
    "9dfd26b5673f80fecb5ab5996b98b5934c06ceb14426f4eccba5ebf5f56944e9"
)

DEFAULT_SOVEREIGN_REFINANCING_MANIFEST_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "reference" / "sovereign_refinancing_v1.json"
)

_ROOT_KEYS = frozenset(
    {
        "schema_version",
        "manifest_id",
        "evaluated_on",
        "partition_count",
        "definition",
        "core_sovereign_issuers",
        "comparison_entities",
        "scope_rules",
        "partitions",
    }
)
_PARTITION_KEYS = frozenset(
    {
        "partition_id",
        "entity",
        "entity_kind",
        "family",
        "phase",
        "publisher",
        "native_identity",
        "source_url",
    }
)
_ENTITY_KINDS = frozenset(
    {
        "sovereign_issuer",
        "comparison_sovereign",
        "comparison_aggregate",
        "related_subsovereign_system",
    }
)
_FAMILIES = frozenset(
    {
        "maturity_schedule",
        "portfolio_risk_history",
        "funding_plan_outturn",
    }
)
_PHASES = frozenset({"harmonized_scalar", "national_native"})
_OFFICIAL_HOSTS = frozenset(
    {
        "api.fiscaldata.treasury.gov",
        "data-api.ecb.europa.eu",
        "dea.gov.in",
        "ec.europa.eu",
        "www.aft.gouv.fr",
        "www.deutsche-finanzagentur.de",
        "www.dmo.gov.uk",
        "www.dt.mef.gov.it",
        "www.mof.go.jp",
        "www.riksgalden.se",
        "www.tesourotransparente.gov.br",
        "yss.mof.gov.cn",
        "zwgls.mof.gov.cn",
    }
)
_PARTITION_ID = re.compile(r"[a-z][a-z0-9_]{2,95}")


@dataclass(frozen=True)
class SovereignRefinancingPartition:
    """One logical, independently refreshed evidence stream."""

    partition_id: str
    entity: str
    entity_kind: str
    family: str
    phase: str
    publisher: str
    native_identity: str
    source_url: str


@dataclass(frozen=True)
class SovereignRefinancingManifest:
    """Validated immutable package denominator."""

    schema_version: int
    manifest_id: str
    evaluated_on: date
    definition: str
    core_sovereign_issuers: tuple[str, ...]
    comparison_entities: tuple[str, ...]
    scope_rules: tuple[str, ...]
    partitions: tuple[SovereignRefinancingPartition, ...]
    semantic_sha256: str
    file_sha256: str


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _required_string(value: object, field: str, *, max_length: int = 512) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{field} must be a non-empty trimmed string")
    if len(value) > max_length:
        raise ValueError(f"{field} exceeds {max_length} characters")
    return value


def _string_tuple(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty list")
    items = tuple(_required_string(item, f"{field}[]", max_length=512) for item in value)
    if len(items) != len(set(items)):
        raise ValueError(f"{field} contains duplicates")
    return items


def _partition(value: object, position: int) -> SovereignRefinancingPartition:
    if not isinstance(value, dict) or set(value) != _PARTITION_KEYS:
        keys = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise ValueError(f"partitions[{position}] has invalid keys: {keys}")
    partition_id = _required_string(value["partition_id"], "partition_id", max_length=96)
    if _PARTITION_ID.fullmatch(partition_id) is None:
        raise ValueError(f"invalid refinancing partition_id: {partition_id!r}")
    entity = _required_string(value["entity"], "entity", max_length=16)
    entity_kind = _required_string(value["entity_kind"], "entity_kind", max_length=40)
    family = _required_string(value["family"], "family", max_length=40)
    phase = _required_string(value["phase"], "phase", max_length=32)
    if entity_kind not in _ENTITY_KINDS:
        raise ValueError(f"unsupported entity_kind for {partition_id}: {entity_kind!r}")
    if family not in _FAMILIES:
        raise ValueError(f"unsupported family for {partition_id}: {family!r}")
    if phase not in _PHASES:
        raise ValueError(f"unsupported phase for {partition_id}: {phase!r}")
    publisher = _required_string(value["publisher"], "publisher", max_length=160)
    native_identity = _required_string(value["native_identity"], "native_identity", max_length=256)
    source_url = _required_string(value["source_url"], "source_url", max_length=1024)
    parsed = urlsplit(source_url)
    if (
        parsed.scheme != "https"
        or parsed.hostname not in _OFFICIAL_HOSTS
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port is not None
        or parsed.fragment
    ):
        raise ValueError(f"invalid official source_url for {partition_id}: {source_url!r}")
    return SovereignRefinancingPartition(
        partition_id=partition_id,
        entity=entity,
        entity_kind=entity_kind,
        family=family,
        phase=phase,
        publisher=publisher,
        native_identity=native_identity,
        source_url=source_url,
    )


def _semantic_payload(
    *,
    schema_version: int,
    manifest_id: str,
    evaluated_on: date,
    definition: str,
    core_sovereign_issuers: tuple[str, ...],
    comparison_entities: tuple[str, ...],
    scope_rules: tuple[str, ...],
    partitions: tuple[SovereignRefinancingPartition, ...],
) -> dict[str, object]:
    return {
        "schema_version": schema_version,
        "manifest_id": manifest_id,
        "evaluated_on": evaluated_on.isoformat(),
        "partition_count": len(partitions),
        "definition": definition,
        "core_sovereign_issuers": list(core_sovereign_issuers),
        "comparison_entities": list(comparison_entities),
        "scope_rules": list(scope_rules),
        "partitions": [
            {
                "partition_id": item.partition_id,
                "entity": item.entity,
                "entity_kind": item.entity_kind,
                "family": item.family,
                "phase": item.phase,
                "publisher": item.publisher,
                "native_identity": item.native_identity,
                "source_url": item.source_url,
            }
            for item in partitions
        ],
    }


def load_sovereign_refinancing_manifest(
    path: Path = DEFAULT_SOVEREIGN_REFINANCING_MANIFEST_PATH,
) -> SovereignRefinancingManifest:
    """Load and fail closed on any denominator or source-boundary drift."""

    try:
        raw_bytes = path.read_bytes()
        root = json.loads(raw_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load sovereign refinancing manifest: {path}") from exc
    if not isinstance(root, dict) or set(root) != _ROOT_KEYS:
        keys = sorted(root) if isinstance(root, dict) else type(root).__name__
        raise ValueError(f"sovereign refinancing manifest has invalid keys: {keys}")

    schema_version = root["schema_version"]
    if schema_version != SOVEREIGN_REFINANCING_MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported sovereign refinancing manifest schema_version")
    manifest_id = _required_string(root["manifest_id"], "manifest_id", max_length=64)
    if manifest_id != SOVEREIGN_REFINANCING_MANIFEST_ID:
        raise ValueError("unexpected sovereign refinancing manifest_id")
    try:
        evaluated_on = date.fromisoformat(_required_string(root["evaluated_on"], "evaluated_on"))
    except ValueError as exc:
        raise ValueError("evaluated_on must be an ISO calendar date") from exc
    definition = _required_string(root["definition"], "definition", max_length=512)
    core = _string_tuple(root["core_sovereign_issuers"], "core_sovereign_issuers")
    comparison = _string_tuple(root["comparison_entities"], "comparison_entities")
    scope_rules = _string_tuple(root["scope_rules"], "scope_rules")
    if tuple(sorted(core)) != core:
        raise ValueError("core_sovereign_issuers must be sorted")
    if set(core) & set(comparison):
        raise ValueError("core and comparison entity sets must not overlap")

    raw_partitions = root["partitions"]
    if not isinstance(raw_partitions, list):
        raise ValueError("partitions must be a list")
    partitions = tuple(_partition(item, i) for i, item in enumerate(raw_partitions))
    declared_count = root["partition_count"]
    if isinstance(declared_count, bool) or declared_count != len(partitions):
        raise ValueError("partition_count does not match partitions")
    if len(partitions) != SOVEREIGN_REFINANCING_PARTITION_COUNT:
        raise ValueError("sovereign refinancing denominator must contain exactly 48 partitions")
    ids = tuple(item.partition_id for item in partitions)
    if len(ids) != len(set(ids)):
        raise ValueError("sovereign refinancing partition IDs must be unique")
    identities = tuple((item.publisher, item.native_identity) for item in partitions)
    if len(identities) != len(set(identities)):
        raise ValueError("sovereign refinancing native identities must be unique per publisher")
    scalar_count = sum(item.phase == "harmonized_scalar" for item in partitions)
    native_count = sum(item.phase == "national_native" for item in partitions)
    if (
        scalar_count != SOVEREIGN_REFINANCING_SCALAR_PARTITION_COUNT
        or native_count != SOVEREIGN_REFINANCING_NATIVE_PARTITION_COUNT
    ):
        raise ValueError("sovereign refinancing phase counts must be exactly 31 and 17")
    issuer_entities = {item.entity for item in partitions if item.entity_kind == "sovereign_issuer"}
    if issuer_entities != set(core):
        raise ValueError("manifest sovereign issuers do not match the declared core set")

    payload = _semantic_payload(
        schema_version=schema_version,
        manifest_id=manifest_id,
        evaluated_on=evaluated_on,
        definition=definition,
        core_sovereign_issuers=core,
        comparison_entities=comparison,
        scope_rules=scope_rules,
        partitions=partitions,
    )
    return SovereignRefinancingManifest(
        schema_version=schema_version,
        manifest_id=manifest_id,
        evaluated_on=evaluated_on,
        definition=definition,
        core_sovereign_issuers=core,
        comparison_entities=comparison,
        scope_rules=scope_rules,
        partitions=partitions,
        semantic_sha256=hashlib.sha256(_canonical_json_bytes(payload)).hexdigest(),
        file_sha256=hashlib.sha256(raw_bytes).hexdigest(),
    )


def sovereign_refinancing_catalogue_sha256(
    path: Path = DEFAULT_SOVEREIGN_REFINANCING_MANIFEST_PATH,
) -> str:
    """Return the checked denominator's semantic identity."""

    return load_sovereign_refinancing_manifest(path).semantic_sha256


def load_checked_sovereign_refinancing_manifest(
    path: Path = DEFAULT_SOVEREIGN_REFINANCING_MANIFEST_PATH,
) -> SovereignRefinancingManifest:
    """Load the committed denominator and require its pinned semantic hash."""

    manifest = load_sovereign_refinancing_manifest(path)
    if manifest.semantic_sha256 != SOVEREIGN_REFINANCING_MANIFEST_SHA256:
        raise ValueError("sovereign refinancing manifest semantic SHA-256 is not approved")
    return manifest


__all__ = [
    "DEFAULT_SOVEREIGN_REFINANCING_MANIFEST_PATH",
    "SOVEREIGN_REFINANCING_MANIFEST_ID",
    "SOVEREIGN_REFINANCING_MANIFEST_SHA256",
    "SOVEREIGN_REFINANCING_MANIFEST_SCHEMA_VERSION",
    "SOVEREIGN_REFINANCING_NATIVE_PARTITION_COUNT",
    "SOVEREIGN_REFINANCING_PARTITION_COUNT",
    "SOVEREIGN_REFINANCING_SCALAR_PARTITION_COUNT",
    "SOVEREIGN_REFINANCING_VINTAGE_PREFIX",
    "SovereignRefinancingManifest",
    "SovereignRefinancingPartition",
    "load_sovereign_refinancing_manifest",
    "load_checked_sovereign_refinancing_manifest",
    "sovereign_refinancing_catalogue_sha256",
]
