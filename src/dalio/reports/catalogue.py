"""Strict checked-in catalogue for locally acquired official report issues.

The catalogue records issue-specific provenance. Recurring publisher identity is
authoritative in :mod:`dalio.reports.manifest` and is deliberately derived from
that manifest rather than repeated as editable JSON metadata.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from urllib.parse import urlsplit

from dalio.reports.manifest import REPORT_SOURCES, ReportSourceSpec
from dalio.storage.reports import ReportMeta

CATALOGUE_SCHEMA_VERSION = 1

_TOP_LEVEL_FIELDS = frozenset({"schema_version", "issues"})
_ISSUE_FIELDS = frozenset(
    {
        "source_id",
        "issue_key",
        "title",
        "document_date",
        "published_at",
        "available_at",
        "retrieved_at",
        "issue_url",
        "artifact_url",
        "sha256",
        "page_count",
        "artifact_filename",
    }
)
_ISSUE_KEY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SOURCES_BY_ID = {source.source_id: source for source in REPORT_SOURCES}


@dataclass(frozen=True)
class ReportIssue:
    """One exact, independently verifiable official report artifact."""

    source_id: str
    issue_key: str
    title: str
    document_date: date
    published_at: datetime | None
    available_at: datetime
    retrieved_at: datetime
    issue_url: str
    artifact_url: str
    sha256: str
    page_count: int
    artifact_filename: str

    @property
    def identity(self) -> str:
        return f"{self.source_id}:{self.issue_key}"

    @property
    def source(self) -> ReportSourceSpec:
        return _SOURCES_BY_ID[self.source_id]

    def to_report_meta(self) -> ReportMeta:
        """Translate catalogue provenance into the immutable report-ledger contract."""
        source = self.source
        return ReportMeta(
            source_id=source.source_id,
            report_family=source.report_family,
            issue_key=self.issue_key,
            publisher=source.publisher,
            jurisdiction=source.jurisdiction,
            title=self.title,
            language=source.language,
            document_date=self.document_date,
            published_at=self.published_at,
            available_at=self.available_at,
            retrieved_at=self.retrieved_at,
            landing_url=source.landing_url,
            artifact_url=self.artifact_url,
            allowed_domains=source.official_domains,
            expected_sha256=self.sha256,
        )


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _exact_fields(payload: dict[str, object], expected: frozenset[str], label: str) -> None:
    supplied = set(payload)
    missing = sorted(expected - supplied)
    unknown = sorted(supplied - expected)
    if missing:
        raise ValueError(f"{label} has missing fields: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"{label} has unknown fields: {', '.join(unknown)}")


def _required_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    if value != value.strip():
        raise ValueError(f"{field} must not contain leading or trailing whitespace")
    return value


def _parse_date(value: object, field: str) -> date:
    raw = _required_string(value, field)
    try:
        parsed = date.fromisoformat(raw)
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO YYYY-MM-DD date") from exc
    if raw != parsed.isoformat():
        raise ValueError(f"{field} must be an ISO YYYY-MM-DD date")
    return parsed


def _parse_datetime(value: object, field: str, *, nullable: bool = False) -> datetime | None:
    if value is None and nullable:
        return None
    raw = _required_string(value, field)
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO-8601 datetime") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must include a timezone")
    return parsed.astimezone(UTC)


def _validate_official_url(url: object, source: ReportSourceSpec, field: str) -> str:
    raw = _required_string(url, field)
    parsed = urlsplit(raw)
    if parsed.scheme.lower() != "https" or not parsed.hostname:
        raise ValueError(f"{field} must be an HTTPS URL")
    if parsed.username or parsed.password:
        raise ValueError(f"{field} must not contain user credentials")
    host = parsed.hostname.lower().strip(".")
    if not any(
        host == domain.lower() or host.endswith(f".{domain.lower()}")
        for domain in source.official_domains
    ):
        raise ValueError(f"{field} host {host!r} is not allowlisted for {source.source_id}")
    return raw


def _parse_issue(payload: object, index: int) -> ReportIssue:
    label = f"issues[{index}]"
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be an object")
    _exact_fields(payload, _ISSUE_FIELDS, label)

    source_id = _required_string(payload["source_id"], f"{label}.source_id")
    source = _SOURCES_BY_ID.get(source_id)
    if source is None:
        raise ValueError(f"{label}.source_id is not in the official report manifest: {source_id}")
    if not source.enabled:
        raise ValueError(f"{label}.source_id is disabled in the official report manifest")

    issue_key = _required_string(payload["issue_key"], f"{label}.issue_key")
    if _ISSUE_KEY_RE.fullmatch(issue_key) is None:
        raise ValueError(f"{label}.issue_key contains unsupported characters")
    title = _required_string(payload["title"], f"{label}.title")
    document_date = _parse_date(payload["document_date"], f"{label}.document_date")
    published_at = _parse_datetime(payload["published_at"], f"{label}.published_at", nullable=True)
    available_at = _parse_datetime(payload["available_at"], f"{label}.available_at")
    retrieved_at = _parse_datetime(payload["retrieved_at"], f"{label}.retrieved_at")
    assert available_at is not None and retrieved_at is not None
    if published_at is not None and available_at < published_at:
        raise ValueError(f"{label}.available_at cannot be earlier than published_at")
    if retrieved_at < available_at:
        raise ValueError(f"{label}.retrieved_at cannot be earlier than available_at")
    if document_date > available_at.date():
        raise ValueError(f"{label}.document_date cannot be later than available_at")

    sha256 = _required_string(payload["sha256"], f"{label}.sha256")
    if _SHA256_RE.fullmatch(sha256) is None:
        raise ValueError(f"{label}.sha256 must be 64 lowercase hexadecimal characters")
    page_count = payload["page_count"]
    if isinstance(page_count, bool) or not isinstance(page_count, int) or page_count < 1:
        raise ValueError(f"{label}.page_count must be a positive integer")
    filename = _required_string(payload["artifact_filename"], f"{label}.artifact_filename")
    if (
        filename in {".", ".."}
        or Path(filename).name != filename
        or "/" in filename
        or "\\" in filename
        or Path(filename).suffix.lower() != ".pdf"
    ):
        raise ValueError(f"{label}.artifact_filename must be a basename ending in .pdf")

    return ReportIssue(
        source_id=source_id,
        issue_key=issue_key,
        title=title,
        document_date=document_date,
        published_at=published_at,
        available_at=available_at,
        retrieved_at=retrieved_at,
        issue_url=_validate_official_url(payload["issue_url"], source, f"{label}.issue_url"),
        artifact_url=_validate_official_url(
            payload["artifact_url"], source, f"{label}.artifact_url"
        ),
        sha256=sha256,
        page_count=page_count,
        artifact_filename=filename,
    )


def load_issue_catalogue(path: Path) -> tuple[ReportIssue, ...]:
    """Load a schema-exact issue catalogue, rejecting unsafe ambiguity."""
    source_path = Path(path)
    try:
        payload = json.loads(
            source_path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_json_object,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read report issue catalogue {source_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("report issue catalogue must be a JSON object")
    _exact_fields(payload, _TOP_LEVEL_FIELDS, "report issue catalogue")
    if (
        isinstance(payload["schema_version"], bool)
        or not isinstance(payload["schema_version"], int)
        or payload["schema_version"] != CATALOGUE_SCHEMA_VERSION
    ):
        raise ValueError(
            f"report issue catalogue schema_version must be {CATALOGUE_SCHEMA_VERSION}"
        )
    raw_issues = payload["issues"]
    if not isinstance(raw_issues, list) or not raw_issues:
        raise ValueError("report issue catalogue issues must be a non-empty array")
    issues = tuple(_parse_issue(item, index) for index, item in enumerate(raw_issues))

    identities = [issue.identity for issue in issues]
    duplicate_identities = sorted(
        identity for identity in set(identities) if identities.count(identity) > 1
    )
    if duplicate_identities:
        raise ValueError(f"duplicate issue identity: {', '.join(duplicate_identities)}")
    filenames = [issue.artifact_filename.casefold() for issue in issues]
    duplicate_filenames = sorted(
        filename for filename in set(filenames) if filenames.count(filename) > 1
    )
    if duplicate_filenames:
        raise ValueError(f"duplicate artifact_filename: {', '.join(duplicate_filenames)}")
    return issues


__all__ = ["CATALOGUE_SCHEMA_VERSION", "ReportIssue", "load_issue_catalogue"]
