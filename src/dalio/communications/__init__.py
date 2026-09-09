"""Institutional communications metadata and evidence contracts."""

from dalio.communications.catalogue import (
    COMMUNICATION_CATALOGUE_SHA256,
    COMMUNICATION_CATALOGUE_SNAPSHOTS,
    COMMUNICATION_SOURCES,
    CommunicationCatalogueSnapshot,
    CommunicationSourceSpec,
    communication_catalogue_sha256,
    communication_catalogue_snapshot,
    resolve_communication_catalogue_snapshot,
)

__all__ = [
    "COMMUNICATION_CATALOGUE_SHA256",
    "COMMUNICATION_CATALOGUE_SNAPSHOTS",
    "COMMUNICATION_SOURCES",
    "CommunicationCatalogueSnapshot",
    "CommunicationSourceSpec",
    "communication_catalogue_snapshot",
    "communication_catalogue_sha256",
    "resolve_communication_catalogue_snapshot",
]
