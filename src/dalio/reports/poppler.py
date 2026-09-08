"""Deterministic physical-page text extraction through Poppler ``pdftotext``."""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

from dalio.storage.reports import PageText

_VERSION_RE = re.compile(rb"pdftotext version\s+([^\s]+)", re.IGNORECASE)


class PopplerPageExtractor:
    """Extract UTF-8 text while preserving Poppler's physical-page separators.

    The command is a fixed argument vector and never invokes a shell. The
    extractor version records both the Poppler binary version and this adapter's
    output contract so changing either creates a new immutable extraction.
    """

    name = "poppler-pdftotext"

    def __init__(
        self,
        executable: str = "pdftotext",
        *,
        timeout_seconds: int = 180,
        temp_root: Path | None = None,
    ) -> None:
        if not executable.strip():
            raise ValueError("pdftotext executable must not be empty")
        if isinstance(timeout_seconds, bool) or timeout_seconds < 1:
            raise ValueError("timeout_seconds must be positive")
        self.executable = executable
        self.timeout_seconds = timeout_seconds
        self.temp_root = Path(temp_root) if temp_root is not None else None
        binary_version = self._detect_version()
        self.version = f"{binary_version}-layout-utf8-eol-unix-v1"

    def _run(self, command: list[str]) -> subprocess.CompletedProcess[bytes]:
        try:
            result = subprocess.run(
                command,
                input=None,
                capture_output=True,
                timeout=self.timeout_seconds,
                check=False,
                shell=False,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(f"pdftotext executable was not found: {self.executable}") from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("pdftotext timed out") from exc
        if result.returncode != 0:
            detail = result.stderr.decode("utf-8", errors="replace").strip()
            suffix = f": {detail[:500]}" if detail else ""
            raise RuntimeError(f"pdftotext failed with exit code {result.returncode}{suffix}")
        return result

    def _detect_version(self) -> str:
        result = self._run([self.executable, "-v"])
        output = result.stderr + b"\n" + result.stdout
        match = _VERSION_RE.search(output)
        if match is None:
            raise RuntimeError("could not determine pdftotext version")
        try:
            return match.group(1).decode("ascii")
        except UnicodeDecodeError as exc:
            raise RuntimeError("pdftotext reported a non-ASCII version") from exc

    def extract(self, pdf_bytes: bytes) -> tuple[PageText, ...]:
        if not isinstance(pdf_bytes, bytes) or not pdf_bytes.startswith(b"%PDF-"):
            raise ValueError("report bytes must have a PDF %PDF- magic header")
        with tempfile.TemporaryDirectory(prefix="dalio-report-", dir=self.temp_root) as temp_dir:
            input_path = Path(temp_dir) / "source.pdf"
            input_path.write_bytes(pdf_bytes)
            result = self._run(
                [
                    self.executable,
                    "-layout",
                    "-enc",
                    "UTF-8",
                    "-eol",
                    "unix",
                    "-q",
                    str(input_path),
                    "-",
                ]
            )
        try:
            text = result.stdout.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise RuntimeError("pdftotext output was not valid UTF-8") from exc
        chunks = text.split("\f")
        # Poppler terminates each physical page with form feed. Discard only the
        # segment after the final delimiter; an actual blank page remains.
        if len(chunks) > 1 and not chunks[-1].strip():
            chunks.pop()
        if not chunks or (len(chunks) == 1 and chunks[0] == ""):
            raise RuntimeError("pdftotext returned no physical pages")
        pages = []
        for number, chunk in enumerate(chunks, start=1):
            normalized = chunk.replace("\r\n", "\n").replace("\r", "\n").rstrip("\n")
            pages.append(PageText(pdf_page=number, text=normalized))
        return tuple(pages)


__all__ = ["PopplerPageExtractor"]
