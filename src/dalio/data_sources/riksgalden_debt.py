"""Original Riksgälden debt statistics with explicit instrument and forecast scopes.

This bounded contract reads the eight 2026 monthly reports and the 2026:1
borrowing workbook. It does not turn refixing statistics into residual maturity,
backcast today's securities, or treat a borrowing plan as a realised cash flow.
"""

from __future__ import annotations

import calendar
import io
import math
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from zipfile import BadZipFile, ZipFile


@dataclass(frozen=True)
class NationalDebtDocument:
    stream_id: str
    snapshot_key: str
    country: str
    source_url: str
    published_at: datetime
    reference_date: date
    source_bytes: bytes
    facts: tuple[dict, ...]
    metadata: dict


@dataclass(frozen=True)
class RiksgaldenDocumentSpec:
    stream_id: str
    snapshot_key: str
    source_url: str
    published_at: datetime
    reference_date: date


_MONTHLY_STREAM = "se_central_government_debt_monthly_report"
_FUNDING_STREAM = "se_central_government_funding_plan"
_HOST = "https://www.riksgalden.se/contentassets/"
_MONTHLY_CATALOGUE = (
    ("01-30", "02-06", "b5235839ebe54707a16c7fe6925635ad", "jan"),
    ("02-27", "03-06", "fed24c5da3454fd1b8230f661cf8bc3b", "feb"),
    ("03-31", "04-09", "894b4829c97342c79f0311245da01114", "mar"),
    ("04-30", "05-08", "c26f6e27b6bf47088d4ceb66701c3944", "apr"),
    ("05-29", "06-05", "4521794aa9cb40238609ee966ea9fba3", "may"),
    ("06-30", "07-07", "655c4484c015484abc95576aa1a4b476", "june"),
    ("07-31", "08-07", "5c06a3c321364119a20a775d75f2511b", "july"),
    ("08-31", "09-07", "5e917f0edef443a4be3755bdea15d334", "aug"),
)
MONTHLY_REPORTS_2026 = tuple(
    RiksgaldenDocumentSpec(
        _MONTHLY_STREAM,
        f"2026-{reference}",
        f"{_HOST}{asset}/central-government-debt-{month}-2026.pdf",
        datetime.fromisoformat(f"2026-{published}T00:00:00+00:00"),
        date.fromisoformat(f"2026-{reference}"),
    )
    for reference, published, asset, month in _MONTHLY_CATALOGUE
)
FUNDING_PLAN_2026_1 = RiksgaldenDocumentSpec(
    _FUNDING_STREAM,
    "2026-1",
    f"{_HOST}27fcc0e426c54e508344a41630b84c99/central_government_borrowing-2026-1-data.xlsx",
    datetime(2026, 5, 28, tzinfo=UTC),
    date(2026, 5, 13),  # Report's explicit information cut-off, not a stock date.
)
RIKSGALDEN_DOCUMENT_SPECS = (*MONTHLY_REPORTS_2026, FUNDING_PLAN_2026_1)
SCHEMA_VERSION = "riksgalden-debt-native-v1"
MONTHLY_SOURCE_INDEX_URL = (
    "https://www.riksgalden.se/en/press-and-publications/publications/Government-debt/"
)
MONTHLY_SOURCE_INDEX_SHA256 = "f329deb4266ce1877ff53829e55843ab3a1daf4e9e6c1308c9698a42c325cd05"
FUNDING_SOURCE_INDEX_URL = (
    "https://www.riksgalden.se/en/press-and-publications/publications/government-borrowing/"
)


def _check_spec(spec: RiksgaldenDocumentSpec, stream: str) -> None:
    if spec not in RIKSGALDEN_DOCUMENT_SPECS or spec.stream_id != stream:
        raise ValueError("Unknown or modified original Riksgalden document contract")


def _number(value: str) -> float:
    if not re.fullmatch(r"-?(?:\d+|\d{1,3}(?: \d{3})+)(?:[.,]\d+)?", value):
        raise ValueError(f"Invalid native numeric field: {value!r}")
    result = float(value.replace(" ", "").replace(",", "."))
    if not math.isfinite(result):
        raise ValueError("Nonfinite native value")
    return result


def _fact(kind, metric, value, unit, start, end, status, dimensions, locator, label, raw):
    if value is not None and not math.isfinite(value):
        raise ValueError("Nonfinite fact")
    return {
        "fact_type": kind,
        "metric": metric,
        "value": value,
        "unit": unit,
        "period_start": start,
        "period_end": end,
        "status": status,
        "dimensions": dimensions,
        "source_locator": locator,
        "native_label": label,
        "native_value": raw,
    }


def _document(spec, body, facts, metadata):
    return NationalDebtDocument(
        spec.stream_id,
        spec.snapshot_key,
        "SE",
        spec.source_url,
        spec.published_at,
        spec.reference_date,
        body,
        tuple(facts),
        {
            "schema_version": SCHEMA_VERSION,
            "publisher": "Swedish National Debt Office",
            "publication_precision": "date",
            **metadata,
        },
    )


def _pdf_layout_text(body: bytes) -> str:
    if not body.startswith(b"%PDF-") or len(body) > 10_000_000:
        raise ValueError("Expected a bounded PDF source file")
    with tempfile.TemporaryDirectory(prefix="dalio-riksgalden-") as folder:
        path = Path(folder) / "source.pdf"
        path.write_bytes(body)
        try:
            result = subprocess.run(
                ["pdftotext", "-layout", "-enc", "UTF-8", str(path), "-"],
                check=True,
                capture_output=True,
                timeout=30,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise ValueError("Cannot extract original PDF with pdftotext") from exc
    return result.stdout.decode("utf-8")


def _columns(line: str) -> list[str]:
    return re.split(r"\s{2,}", line.strip())


def _tail_reconciles(line: str, amounts: list[float]) -> None:
    """Match exact integer totals, including adjacent PDF columns a space apart."""
    rest = line.rstrip()
    for i, value in enumerate(reversed(amounts)):
        # The PDF rounds each security and its independently calculated subtotal
        # to whole SEK; validated source files differ by at most one SEK.
        tolerance = (0,) if i < 2 else (0, -1, 1)
        tokens = [f"{int(value) + delta:,}".replace(",", " ") for delta in tolerance]
        candidates = [token for token in tokens if rest.endswith(token)]
        if len(candidates) != 1:
            raise ValueError("Security table does not reconcile to printed totals")
        token = candidates[0]
        rest = rest[: -len(token)]
        if rest and not rest[-1].isspace():
            raise ValueError("Ambiguous PDF subtotal boundary")
        rest = rest.rstrip()
    if not re.fullmatch(r"-?\d+[,.]\d+|-", rest.strip()):
        raise ValueError("Unexpected security subtotal columns")


def _section(page: str, start: str, end: str) -> str:
    first = list(re.finditer(r"^" + re.escape(start) + r"(?=\s|$)", page, re.M))
    last = list(re.finditer(r"^" + re.escape(end) + r"(?=\s|$)", page, re.M))
    if len(first) != 1 or len(last) != 1 or first[0].end() >= last[0].start():
        raise ValueError(f"Missing or ambiguous PDF section {start}")
    return page[first[0].end() : last[0].start()]


def parse_monthly_report(body: bytes, *, spec: RiksgaldenDocumentSpec) -> NationalDebtDocument:
    _check_spec(spec, _MONTHLY_STREAM)
    if not body.startswith(b"%PDF-"):
        raise ValueError("Expected PDF bytes")
    pages = _pdf_layout_text(body).split("\f")
    if len(pages) != 11 or pages[-1].strip():
        raise ValueError("Expected ten-page monthly statistical report")
    if spec.reference_date.strftime("%-d %B %Y") not in pages[0]:
        raise ValueError("PDF reference date differs from pinned source")
    required = (
        (0, "Official measure of the central government's gross debt:"),
        (0, "Liabilities are reported with a positive sign and assets with a negative."),
        (1, "Time to"),
        (1, "Refixing"),
        (3, "Different debt classes, including on-lending and assets under management"),
        (9, "AVERAGE TIME TO REFIXING AS MEASURED IN RISK MANAGEMENT"),
        (9, "The maturity is measured with Average Time to Refixing."),
        (9, "monthly average"),
    )
    if any(label not in pages[p] for p, label in required):
        raise ValueError("PDF accounting or maturity method changed")
    ref = spec.reference_date
    month_start = ref.replace(day=1)
    month_end = ref.replace(day=calendar.monthrange(ref.year, ref.month)[1])
    facts = []
    levels = {}
    level_specs = (
        ("A. Nominal amount, incl.", "nominal_debt_including_assets_sek"),
        ("Accrued inflation compensation", "accrued_inflation_compensation_sek"),
        ("Exchange rate effect", "exchange_rate_effect_sek"),
        ("B. Nominal uplifted amount", "uplifted_debt_including_assets_sek"),
        ("Assets under management, current exchange rate", "assets_added_for_gross_debt_sek"),
        ("C. CENTRAL GOVERNMENT DEBT", "central_gov_gross_debt_sek"),
        ("On-lendning", "on_lending_claims_sek"),
        ("D. CENTRAL GOVERNMENT DEBT INCLUDING", "debt_including_on_lending_and_assets_sek"),
    )
    lines = pages[0].splitlines()
    for label, metric in level_specs:
        matches = [i for i, line in enumerate(lines) if line.startswith(label)]
        if len(matches) != 1:
            raise ValueError(f"Missing or duplicated headline {label}")
        pos = matches[0]
        for end in range(pos, min(pos + 3, len(lines))):
            cols = _columns(lines[end])
            try:
                change, value = map(_number, cols[-2:])
            except (ValueError, TypeError):
                continue
            break
        else:
            raise ValueError("Missing headline amount columns")
        levels[metric] = value
        dims = {
            "scope": "central_government",
            "accounting_basis": "business_date",
            "measure_label": label,
            "aggregation": "reference_date",
        }
        facts.append(
            _fact(
                "debt_stock",
                metric,
                value,
                "SEK",
                ref,
                ref,
                "observed",
                dims,
                f"PDF page 1 row {end + 1} outstanding",
                label,
                cols[-1],
            )
        )
        facts.append(
            _fact(
                "debt_change",
                metric.removesuffix("_sek") + "_monthly_change_sek",
                change,
                "SEK",
                month_start,
                ref,
                "observed",
                dims,
                f"PDF page 1 row {end + 1} change",
                label,
                cols[-2],
            )
        )
    a, inflation, fx, b, assets, c, lending, d = levels.values()
    if any(abs(x) > 1 for x in (b - a - inflation - fx, c - b - assets, d - b - lending)) or c <= 0:
        raise ValueError("Headline debt measures do not reconcile")

    # The outstanding inventory is a point-in-time list, with explicit maturity dates.
    # ATR is a separate printed risk statistic, never used to infer principal dates.
    groups = (
        (1, "Government bonds", "Inflation-linked bonds", "government_bonds", 3),
        (1, "Inflation-linked bonds", "Green bonds", "inflation_linked_bonds", 5),
        (1, "Green bonds", "Public bonds in foreign currencies", "green_bonds", 3),
        (
            1,
            "Public bonds in foreign currencies",
            "Private placements",
            "foreign_currency_bonds",
            5,
        ),
        (2, "T-bills", "Liquidity management instruments", "treasury_bills", 3),
    )
    section_totals = {}
    for p, start, end, instrument_class, count in groups:
        section = _section(pages[p], start, end)
        section_lines = section.splitlines()
        rows = [
            (i, line)
            for i, line in enumerate(section_lines)
            if re.match(r"^(SGB|EUB|ESB|STB) ", line)
        ]
        if not rows:
            raise ValueError(f"Missing securities for {instrument_class}")
        sums = [0.0] * (count - 1)
        seen = set()
        for rownum, line in rows:
            match = re.fullmatch(r"(.+?)\s+(\d{4}-\d{2}-\d{2})\s+(.+)", line)
            if match is None:
                raise ValueError("Malformed security row")
            label, issued, numeric = match.groups()
            if label in seen:
                raise ValueError("Duplicated security")
            seen.add(label)
            # A zero uplift next to a negative redemption can lose the second gap.
            numeric = re.sub(r"(?<=\d) (?=-\d)", "  ", numeric)
            cols = _columns(numeric)
            if len(cols) != count:
                raise ValueError(f"Changed security columns: {label}")
            values = [None if x == "-" else _number(x) for x in cols]
            if any(x is None for x in values[1:]):
                raise ValueError("Missing security nominal amount")
            maturity_match = re.search(r"(\d{1,2} [A-Za-z]{3} \d{2})$", label)
            if maturity_match is None:
                raise ValueError("Missing explicit security maturity date")
            day, mon, year = maturity_match.group(1).split()
            maturity = datetime.strptime(f"{day} {mon} 20{year}", "%d %b %Y").date()
            issue = date.fromisoformat(issued)
            if maturity < issue or (maturity < ref and values[-1] != 0):
                raise ValueError("Inconsistent issue/maturity/reference dates")
            if values[-1] < 0:
                raise ValueError("Negative outstanding security amount")
            for i, value in enumerate(values[1:]):
                sums[i] += value
            dimensions = {
                "scope": "central_government_issued_security",
                "instrument_class": instrument_class,
                "instrument_label": label,
                "issue_date": issued,
                "maturity_date": maturity.isoformat(),
                "date_basis": "settlement_date",
                "nominal_sek_amount_basis": "face_value_at_issue_exchange_rate",
                "aggregation": "reference_date",
                "inventory_reference_date": ref.isoformat(),
                "issue_settlement_after_reference_date": issue > ref,
            }
            currencies = re.search(r"(?:EUB|ESB) (USD|EUR) ", label)
            dimensions["issue_currency"] = currencies.group(1) if currencies else "SEK"
            metrics = [("average_time_to_refixing", "years")]
            if instrument_class == "inflation_linked_bonds":
                metrics += [("inflation_compensation", "SEK"), ("uplifted_amount", "SEK")]
                if values[2] != values[-1] + values[1]:
                    raise ValueError("Inflation-linked security uplift does not reconcile")
            if instrument_class == "foreign_currency_bonds":
                metrics += [
                    ("nominal_amount_issue_currency", dimensions["issue_currency"]),
                    ("amount_at_current_exchange_rate", "SEK"),
                ]
            metrics += [("nominal_change", "SEK"), ("nominal_amount", "SEK")]
            for i, ((metric, unit), value) in enumerate(zip(metrics, values, strict=True)):
                kind = "security_change" if metric == "nominal_change" else "security_stock"
                facts.append(
                    _fact(
                        kind,
                        metric,
                        value,
                        unit,
                        month_start if metric == "nominal_change" else ref,
                        ref,
                        "not_reported" if value is None else "observed",
                        dimensions,
                        f"PDF page {p + 1} {start} row {rownum + 1} column {i + 1}",
                        label,
                        cols[i],
                    )
                )
        last = rows[-1][0]
        totals = next((line for line in section_lines[last + 1 :] if line.strip()), None)
        if totals is None:
            raise ValueError("Missing security section total")
        # Unlike amounts in a single foreign currency, a mixed-currency nominal sum
        # is not a published subtotal. Its SEK counterpart is separately checked.
        printed_sums = sums[1:] if instrument_class == "foreign_currency_bonds" else sums
        _tail_reconciles(totals, printed_sums)
        section_totals[instrument_class] = sums[-1]
        facts.append(
            _fact(
                "instrument_total",
                "nominal_amount",
                sums[-1],
                "SEK",
                ref,
                ref,
                "observed",
                {"instrument_class": instrument_class, "scope": "central_government"},
                f"PDF page {p + 1} {start} total",
                start,
                f"{int(sums[-1]):,}".replace(",", " "),
            )
        )
    private = _section(
        pages[1], "Private placements in foreign currencies etc.", "Sum: Capital market"
    )
    if any(re.search(r"\d", line) for line in private.splitlines()):
        raise ValueError("New private-placement entries need their own explicit contract")
    capital_line = next(
        line for line in pages[1].splitlines() if line.startswith("Sum: Capital market")
    )
    if _number(_columns(capital_line)[-1]) != sum(
        v for k, v in section_totals.items() if k != "treasury_bills"
    ):
        raise ValueError("Capital-market inventory does not reconcile")

    for p, aggregation in ((3, "reference_date"), (9, "monthly_mean")):
        section = pages[p].split("Nominal krona debt", 1)[1]
        first_lines = ("Nominal krona debt" + section).splitlines()[:4]
        if len(first_lines) != 4:
            raise ValueError("Missing debt class table")
        class_values = []
        expected = ("Nominal krona debt", "Inflation-linked debt", "Foreign currency debt", "total")
        for i, (line, label) in enumerate(zip(first_lines, expected, strict=True)):
            cols = _columns(line)
            if i < 3:
                if cols[0] != label:
                    raise ValueError("Changed debt class order")
                cols = cols[1:]
            if len(cols) != (4 if p == 3 else 2):
                raise ValueError("Changed debt-class columns")
            vals = list(map(_number, cols))
            class_values.append(vals)
            metrics = (
                [
                    ("nominal_amount", "SEK"),
                    ("uplifted_amount_current_fx", "SEK"),
                    ("average_time_to_refixing", "years"),
                    ("share", "percent"),
                ]
                if p == 3
                else [
                    ("risk_management_debt_measure", "SEK"),
                    ("average_time_to_refixing", "years"),
                ]
            )
            dimensions = {
                "debt_class": label if i < 3 else "total",
                "aggregation": aggregation,
                "scope": "central_government_including_on_lending_and_assets",
                "derivative_basis": (
                    "debt_class_allocation_derivatives_included_active_positions_excluded"
                    if p == 9
                    else "native_debt_class_accounting_including_foreign_exchange_forwards"
                ),
            }
            for n, ((metric, unit), value) in enumerate(zip(metrics, vals, strict=True)):
                facts.append(
                    _fact(
                        "portfolio_risk",
                        metric,
                        value,
                        unit,
                        month_start if p == 9 else ref,
                        month_end if p == 9 else ref,
                        "observed",
                        dimensions,
                        f"PDF page {p + 1} debt class {label} column {n + 1}",
                        label,
                        cols[n],
                    )
                )
        for col in [0, 1] if p == 3 else [0]:
            if abs(sum(row[col] for row in class_values[:3]) - class_values[3][col]) > 1:
                raise ValueError("Debt-class amounts do not reconcile")
        if p == 3 and class_values[3][1] != d:
            raise ValueError("Portfolio measure differs from headline D")
    return _document(
        spec,
        body,
        facts,
        {
            "scope": "central_government",
            "reference_date_basis": "last_reporting_business_day",
            "covered_pdf_pages": [1, 2, 3, 4, 10],
            "covered_sections": [
                "headline_debt_measures",
                "bond_and_bill_inventory",
                "debt_class_balances",
                "risk_management_atr",
            ],
            "unparsed_sections": [
                "liquidity_details",
                "swaps",
                "historical_charts",
                "on_lending_details",
            ],
            "maturity_basis": "average_time_to_refixing_separate_from_security_contractual_maturity",
            "snapshot_revision_policy": "original_publication_methods; workbook and online histories may be revised",
            "headline_reconciliation_tolerance_sek": 1,
            "source_index_url": MONTHLY_SOURCE_INDEX_URL,
            "source_index_sha256": MONTHLY_SOURCE_INDEX_SHA256,
        },
    )


_NS = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
_REL_ID = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
_HEADERS = {
    "F10": (
        "Date",
        "Net borrowing requirement",
        "Redemptions",
        "Other",
        "Gross borrowing requirement, outcome and forecast",
        "Forecast",
        "Gross borrowing requirement, previous forecast",
    ),
    "F11": (
        "Date",
        "Nominal government bonds",
        "Inflation-linked bonds",
        "Foreign currency bonds etc.",
        "Green bonds",
        "On-lending",
        "T-bills",
        "Liquidity management etc.",
        "Retail market",
        "Total",
        "Total, forecast",
    ),
    "F17": ("Date", "Duration", "ATR", "ATR, forecast", "Steering interval", "Steering interval2"),
}


def _xlsx_cells(body: bytes) -> tuple[dict[str, dict], dict[str, dict]]:
    if len(body) > 10_000_000:
        raise ValueError("Workbook exceeds bounded source size")
    try:
        with ZipFile(io.BytesIO(body)) as z:
            infos = z.infolist()
            if (
                len({x.filename for x in infos}) != len(infos)
                or sum(x.file_size for x in infos) > 30_000_000
            ):
                raise ValueError("Unsafe or ambiguous workbook archive")

            def xml(name):
                raw = z.read(name)
                if b"<!DOCTYPE" in raw.upper() or b"<!ENTITY" in raw.upper():
                    raise ValueError("Unexpected XML entity declaration")
                return ET.fromstring(raw)

            shared = []
            if "xl/sharedStrings.xml" in z.namelist():
                shared = [
                    "".join(n.itertext()) for n in xml("xl/sharedStrings.xml").findall("x:si", _NS)
                ]
            rels = {n.attrib["Id"]: n.attrib["Target"] for n in xml("xl/_rels/workbook.xml.rels")}
            result, native_values = {}, {}
            for sheet in xml("xl/workbook.xml").findall("x:sheets/x:sheet", _NS):
                name = sheet.attrib["name"]
                if name not in {"Contents", *_HEADERS}:
                    continue
                if name in result:
                    raise ValueError("Duplicated worksheet")
                target = rels[sheet.attrib[_REL_ID]]
                if not re.fullmatch(r"worksheets/sheet\d+\.xml", target):
                    raise ValueError("Unexpected worksheet relationship")
                cells, raw_cells = {}, {}
                for cell in xml("xl/" + target).findall("x:sheetData/x:row/x:c", _NS):
                    key = cell.attrib["r"]
                    if key in cells or cell.find("x:f", _NS) is not None:
                        raise ValueError("Duplicated or formula-valued statistical cell")
                    value = cell.find("x:v", _NS)
                    kind = cell.attrib.get("t", "n")
                    if kind == "inlineStr":
                        value = "".join(cell.find("x:is", _NS).itertext())
                    elif value is None:
                        continue
                    elif kind == "s":
                        value = shared[int(value.text)]
                    elif kind == "n":
                        raw_cells[key] = value.text
                        value = float(value.text)
                        if not math.isfinite(value):
                            raise ValueError("Nonfinite spreadsheet cell")
                    else:
                        raise ValueError("Unsupported statistical cell type")
                    cells[key] = value
                result[name] = cells
                native_values[name] = raw_cells
            return result, native_values
    except (BadZipFile, KeyError, ET.ParseError, IndexError, TypeError) as exc:
        raise ValueError("Invalid original XLSX workbook") from exc


def _excel_date(value) -> date:
    if not isinstance(value, (float, int)) or not math.isfinite(value) or int(value) != value:
        raise ValueError("Invalid Excel chart date")
    return (datetime(1899, 12, 30) + timedelta(days=value)).date()


def parse_funding_workbook(
    body: bytes, *, spec: RiksgaldenDocumentSpec = FUNDING_PLAN_2026_1
) -> NationalDebtDocument:
    _check_spec(spec, _FUNDING_STREAM)
    tables, native_values = _xlsx_cells(body)
    if set(tables) != {"Contents", *_HEADERS}:
        raise ValueError("Missing required funding workbook sheet")
    if (
        tables["Contents"].get("A2")
        != "Central government borrowing – forecast and analysis 2026:1"
    ):
        raise ValueError("Wrong workbook vintage")
    for sheet, headers in _HEADERS.items():
        table = tables[sheet]
        if tuple(table.get(f"{chr(65 + i)}7") for i in range(len(headers))) != headers:
            raise ValueError(f"Changed {sheet} column contract")
        if table.get("A3") != ("Unit: Years" if sheet == "F17" else "Unit: SEK billion"):
            raise ValueError("Changed workbook units")
        if table.get("A5") != "Source: The Debt Office.":
            raise ValueError("Changed original statistical producer")
    required_notes = {
        "F10": [
            "net borrowing requirement is the budget balance with the opposite sign",
            "settlement date",
            "trade date",
        ],
        "F11": [
            "including on-lending and assets under management",
            "stock outstanding at year-end",
        ],
        "F17": [
            "Macaulay duration",
            "average time to refixing (ATR)",
            "last day of each month",
            "monthly mean",
        ],
    }
    for sheet, notes in required_notes.items():
        if any(note not in tables[sheet].get("A4", "") for note in notes):
            raise ValueError("Changed accounting or maturity method note")
    facts = []
    f10_metrics = (
        "net_borrowing_requirement",
        "redemptions",
        "other_financing_adjustment",
        "gross_borrowing_requirement",
        "gross_borrowing_requirement",
        "gross_borrowing_requirement",
    )
    f11_classes = (
        "nominal_government_bonds",
        "inflation_linked_bonds",
        "foreign_currency_bonds",
        "green_bonds",
        "on_lending",
        "treasury_bills",
        "liquidity_management",
        "retail_market",
        "total",
        "total",
    )
    for sheet in _HEADERS:
        table = tables[sheet]
        rows = sorted(
            int(key[1:]) for key in table if re.fullmatch(r"A\d+", key) and int(key[1:]) >= 8
        )
        expected_count = {"F10": 5, "F11": 10, "F17": 61}[sheet]
        if rows != list(range(8, 8 + expected_count)):
            raise ValueError("Truncated or extended source period axis requires review")
        for row in rows:
            native_date = _excel_date(table[f"A{row}"])
            if sheet == "F17":
                offset = row - 8
                year, mon0 = divmod(2022 * 12 + 11 + offset, 12)
                expected = date(year, mon0 + 1, calendar.monthrange(year, mon0 + 1)[1])
                start, end = expected.replace(day=1), expected
                columns = "BCD"
            else:
                year = (2023 if sheet == "F10" else 2018) + row - 8
                expected = date(year, 6, 15) if sheet == "F10" else date(year, 12, 31)
                start, end = date(year, 1, 1), date(year, 12, 31)
                columns = "BCDEFG" if sheet == "F10" else "BCDEFGHIJK"
            if native_date != expected:
                raise ValueError("Changed or duplicated native date axis")
            if sheet == "F10":
                components = [table.get(f"{col}{row}") for col in "BCD"]
                total = table.get(f"{'E' if year <= 2025 else 'F'}{row}")
                if (
                    not all(isinstance(x, (int, float)) for x in [*components, total])
                    or abs(sum(components) - total) > 0.201
                ):
                    raise ValueError("Gross borrowing figures do not reconcile within rounding")
            elif sheet == "F11":
                components = [table.get(f"{col}{row}") for col in "BCDEFGHI"]
                total = table.get(f"{'J' if year <= 2025 else 'K'}{row}")
                if (
                    not all(isinstance(x, (int, float)) for x in [*components, total])
                    or abs(sum(components) - total) > 4.5
                ):
                    raise ValueError("Debt instrument figures do not reconcile within rounding")
            for i, col in enumerate(columns):
                key = f"{col}{row}"
                raw = table.get(key)
                if raw is not None and not isinstance(raw, (int, float)):
                    raise ValueError("Nonnumeric financial field")
                dims = {
                    "native_chart_date": native_date.isoformat(),
                    "native_column": col,
                    "scope": "central_government",
                    "forecast_vintage": "2026-1",
                }
                anchor = False
                if sheet == "F10":
                    metric, kind, unit = f10_metrics[i], "funding_flow", "SEK_bn"
                    status = "forecast" if year > 2025 or col == "G" else "observed"
                    if col == "G":
                        dims["forecast_vintage"] = "2025-2"
                    anchor = col == "F" and year == 2025
                    dims["aggregation"] = "calendar_year"
                    dims["flow_basis"] = (
                        "settlement_date" if col == "B" else "trade_date_with_published_adjustments"
                    )
                    expected_present = (
                        col in "BCD"
                        or (col == "E" and year <= 2025)
                        or (col in "FG" and year >= 2025)
                    )
                elif sheet == "F11":
                    metric, kind, unit = "debt_outstanding", "funding_plan_stock", "SEK_bn"
                    status = "forecast" if year > 2025 else "observed"
                    dims.update(
                        instrument_class=f11_classes[i],
                        aggregation="year_end",
                        scope="central_government_including_on_lending_and_assets",
                    )
                    anchor = col == "K" and year == 2025
                    expected_present = (
                        col in "BCDEFGHI"
                        or (col == "J" and year <= 2025)
                        or (col == "K" and year >= 2025)
                    )
                else:
                    metric = "macaulay_duration" if col == "B" else "average_time_to_refixing"
                    kind, unit = "portfolio_risk", "years"
                    cutoff = date(2026, 4, 30)
                    anchor = col == "D" and end == cutoff
                    status = "forecast" if col == "D" and end > cutoff else "observed"
                    dims.update(
                        aggregation="month_end" if status == "forecast" else "monthly_mean",
                        scope="central_government_risk_management",
                    )
                    expected_present = (
                        (col == "B" and end <= date(2024, 12, 31))
                        or (col == "C" and date(2025, 1, 31) <= end <= cutoff)
                        or (col == "D" and end >= cutoff)
                    )
                    if raw is not None and raw < 0:
                        raise ValueError("Negative total government maturity statistic")
                if expected_present != (raw is not None):
                    raise ValueError(
                        "Missing reported observation or changed historical/forecast boundary"
                    )
                if anchor:
                    prior_col = {"F10": "E", "F11": "J", "F17": "C"}[sheet]
                    if raw != table.get(f"{prior_col}{row}"):
                        raise ValueError("Forecast line anchor differs from observed value")
                    dims["forecast_line_anchor"] = True
                if raw is None:
                    status = "not_reported"
                    dims["missing_reason"] = "structural_blank_in_published_chart_columns"
                facts.append(
                    _fact(
                        kind,
                        metric,
                        raw,
                        unit,
                        start,
                        end,
                        status,
                        dims,
                        f"XLSX {sheet}!{key}",
                        _HEADERS[sheet][ord(col) - 65],
                        "" if raw is None else native_values[sheet][key],
                    )
                )
    return _document(
        spec,
        body,
        facts,
        {
            "covered_sheets": ["F10", "F11", "F17"],
            "reference_date_basis": "report_information_cutoff",
            "forecast_years": [2026, 2027],
            "previous_forecast_vintage": "2025-2",
            "method_break": {
                "old": "macaulay_duration",
                "new": "average_time_to_refixing",
                "first_new_period": "2025-01",
            },
            "annual_chart_dates_are_not_payment_dates": True,
            "forecast_line_anchors_retain_observed_status": True,
            "source_index_url": FUNDING_SOURCE_INDEX_URL,
            "publisher_notes": {sheet: tables[sheet]["A4"] for sheet in _HEADERS},
            "unparsed_sheets": [f"F{i}" for i in range(1, 18) if i not in (10, 11, 17)],
        },
    )


def reparse_document(document: NationalDebtDocument) -> NationalDebtDocument:
    matches = [spec for spec in RIKSGALDEN_DOCUMENT_SPECS if spec.source_url == document.source_url]
    if len(matches) != 1:
        raise ValueError("Unknown original document URL")
    spec = matches[0]
    if (
        document.stream_id,
        document.snapshot_key,
        document.country,
        document.reference_date,
        document.published_at,
    ) != (spec.stream_id, spec.snapshot_key, "SE", spec.reference_date, spec.published_at):
        raise ValueError("Document identity or clock differs from pinned source")
    parser = parse_monthly_report if spec.stream_id == _MONTHLY_STREAM else parse_funding_workbook
    return parser(document.source_bytes, spec=spec)
