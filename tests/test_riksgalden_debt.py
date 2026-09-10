"""Offline statistical contracts: issuer scope, reconciliation, and forecast boundaries."""

import copy
import io
import json
from dataclasses import replace
from datetime import date
from pathlib import Path
from unittest.mock import patch
from xml.sax.saxutils import escape
from zipfile import ZipFile

import pytest

from dalio.data_sources.riksgalden_debt import (
    FUNDING_PLAN_2026_1,
    MONTHLY_REPORTS_2026,
    parse_funding_workbook,
    parse_monthly_report,
    reparse_document,
)

FIXTURES = Path(__file__).parent / "fixtures"


def workbook(cells=None):
    cells = cells or json.loads((FIXTURES / "riksgalden_2026_1_cells.json").read_text())
    out = io.BytesIO()
    ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    rel = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    with ZipFile(out, "w") as z:
        sheets = []
        relations = []
        for i, (name, table) in enumerate(cells.items(), 1):
            sheets.append(f'<sheet name="{name}" sheetId="{i}" r:id="rId{i}"/>')
            relations.append(f'<Relationship Id="rId{i}" Target="worksheets/sheet{i}.xml"/>')
            rows = {}
            for cell, value in table.items():
                row = "".join(c for c in cell if c.isdigit())
                if isinstance(value, str):
                    body = f'<c r="{cell}" t="inlineStr"><is><t>{escape(value)}</t></is></c>'
                else:
                    body = f'<c r="{cell}"><v>{value}</v></c>'
                rows.setdefault(row, []).append(body)
            content = "".join(f'<row r="{n}">{"".join(v)}</row>' for n, v in rows.items())
            z.writestr(
                f"xl/worksheets/sheet{i}.xml",
                f'<worksheet xmlns="{ns}"><sheetData>{content}</sheetData></worksheet>',
            )
        z.writestr(
            "xl/workbook.xml",
            f'<workbook xmlns="{ns}" xmlns:r="{rel}"><sheets>{"".join(sheets)}</sheets></workbook>',
        )
        z.writestr(
            "xl/_rels/workbook.xml.rels", f"<Relationships>{''.join(relations)}</Relationships>"
        )
    return out.getvalue()


def monthly(text=None, spec=None):
    layout = text or (FIXTURES / "riksgalden_aug_2026_layout.txt").read_text()
    with patch("dalio.data_sources.riksgalden_debt._pdf_layout_text", return_value=layout):
        return parse_monthly_report(b"%PDF-1.7\nfixture", spec=spec or MONTHLY_REPORTS_2026[-1])


def cell(document, locator):
    return next(f for f in document.facts if f["source_locator"] == locator)


def test_monthly_reconciles_securities_and_distinguishes_debt_measures():
    doc = monthly()
    assert doc.reference_date == date(2026, 8, 31)
    assert doc.country == "SE"
    assert doc.stream_id == "se_central_government_debt_monthly_report"
    gross = [f for f in doc.facts if f["metric"] == "central_gov_gross_debt_sek"]
    assert len(gross) == 1 and gross[0]["value"] == 1249558872203
    stocks = [
        f
        for f in doc.facts
        if f["fact_type"] == "security_stock" and f["metric"] == "nominal_amount"
    ]
    assert len(stocks) == 31
    assert (
        sum(f["value"] for f in stocks if f["dimensions"]["instrument_class"] == "government_bonds")
        == 778951450000
    )
    expired = next(f for f in stocks if f["dimensions"]["instrument_label"] == "STB 19 Aug 26")
    assert expired["value"] == 0 and expired["dimensions"]["maturity_date"] == "2026-08-19"
    atr = [
        f
        for f in doc.facts
        if f["metric"] == "average_time_to_refixing"
        and f["dimensions"].get("debt_class") == "total"
    ]
    assert {f["dimensions"]["aggregation"]: f["value"] for f in atr} == {
        "reference_date": 5.05,
        "monthly_mean": 4.85,
    }
    assert all("residual_maturity" not in f["metric"] for f in doc.facts)


@pytest.mark.parametrize(
    "before,after",
    [
        ("96 414 000 000", "96 415 000 000"),
        ("1 249 558 872 203", "1 249 558 872 303"),
        ("31 August 2026", "31 July 2026"),
        ("AVERAGE TIME TO REFIXING AS MEASURED IN RISK MANAGEMENT", "UNKNOWN MATURITY MEASURE"),
    ],
)
def test_pdf_rejects_imbalances_dates_and_method_changes(before, after):
    text = (FIXTURES / "riksgalden_aug_2026_layout.txt").read_text().replace(before, after, 1)
    with pytest.raises(ValueError):
        monthly(text)


def test_pinned_document_and_bytes_are_revalidated():
    with pytest.raises(ValueError):
        monthly(spec=replace(MONTHLY_REPORTS_2026[-1], source_url="https://example.com/fake.pdf"))
    with patch(
        "dalio.data_sources.riksgalden_debt._pdf_layout_text",
        return_value=(FIXTURES / "riksgalden_aug_2026_layout.txt").read_text(),
    ):
        doc = monthly()
        modified = replace(doc, facts=({**doc.facts[0], "value": -123.0}, *doc.facts[1:]))
        assert reparse_document(modified).facts == doc.facts
        with pytest.raises(ValueError):
            reparse_document(replace(doc, reference_date=date(2026, 7, 31)))


def test_workbook_preserves_forecast_status_anchor_native_dates_and_blanks():
    doc = parse_funding_workbook(workbook(), spec=FUNDING_PLAN_2026_1)
    assert doc.snapshot_key == "2026-1"
    gross = cell(doc, "XLSX F10!F11")
    assert gross["value"] == 655.6
    assert gross["status"] == "forecast"
    assert gross["period_start"] == date(2026, 1, 1)
    assert gross["period_end"] == date(2026, 12, 31)
    assert gross["dimensions"]["native_chart_date"] == "2026-06-15"
    assert cell(doc, "XLSX F10!F10")["status"] == "observed"
    assert cell(doc, "XLSX F10!F10")["dimensions"]["forecast_line_anchor"] is True
    assert cell(doc, "XLSX F10!G10")["status"] == "forecast"
    assert cell(doc, "XLSX F10!G10")["dimensions"]["forecast_vintage"] == "2025-2"
    assert cell(doc, "XLSX F10!E11")["value"] is None
    assert cell(doc, "XLSX F10!E11")["status"] == "not_reported"
    assert cell(doc, "XLSX F11!F16")["value"] == -11
    assert cell(doc, "XLSX F17!B32")["metric"] == "macaulay_duration"
    assert cell(doc, "XLSX F17!C33")["metric"] == "average_time_to_refixing"
    assert cell(doc, "XLSX F17!C48")["dimensions"]["aggregation"] == "monthly_mean"
    assert cell(doc, "XLSX F17!D49")["dimensions"]["aggregation"] == "month_end"
    assert cell(doc, "XLSX F17!D48")["status"] == "observed"
    assert reparse_document(doc) == doc


@pytest.mark.parametrize(
    "sheet,ref,value",
    [
        ("F10", "C11", 999.0),
        ("F10", "B11", float("nan")),
        ("F10", "A11", 45092.0),
        ("F11", "K16", 2000.0),
        ("F17", "C32", 4.5),
        ("F17", "D49", -1.0),
        ("F17", "A4", "Different method"),
        ("F10", "A3", "Unit: USD billion"),
    ],
)
def test_workbook_fails_closed_on_changed_dimensions_and_bad_numbers(sheet, ref, value):
    cells = copy.deepcopy(json.loads((FIXTURES / "riksgalden_2026_1_cells.json").read_text()))
    cells[sheet][ref] = value
    with pytest.raises(ValueError):
        parse_funding_workbook(workbook(cells), spec=FUNDING_PLAN_2026_1)


def test_missing_expected_sheet_or_source_url_is_rejected():
    cells = json.loads((FIXTURES / "riksgalden_2026_1_cells.json").read_text())
    del cells["F17"]
    with pytest.raises(ValueError):
        parse_funding_workbook(workbook(cells), spec=FUNDING_PLAN_2026_1)
    with pytest.raises(ValueError):
        parse_funding_workbook(
            workbook(),
            spec=replace(FUNDING_PLAN_2026_1, source_url="https://example.com/file.xlsx"),
        )


@pytest.mark.parametrize("month,index", [("apr", 3), ("may", 4), ("july", 6)])
def test_published_edge_cases_keep_trade_and_settlement_dates_and_signed_exposures(month, index):
    doc = monthly(
        (FIXTURES / f"riksgalden_{month}_2026_layout.txt").read_text(), MONTHLY_REPORTS_2026[index]
    )
    if month == "apr":
        fact = next(
            f
            for f in doc.facts
            if f["native_label"] == "STB 19 Aug 26" and f["metric"] == "nominal_amount"
        )
        assert fact["value"] == 10000000000
        assert fact["dimensions"]["issue_date"] == "2026-05-04"
        assert fact["dimensions"]["issue_settlement_after_reference_date"] is True
        assert fact["period_end"] == date(2026, 4, 30)
    elif month == "may":
        fact = next(
            f
            for f in doc.facts
            if f["native_label"] == "SGB IL 3112 0.125% 1 Jun 26"
            and f["metric"] == "nominal_amount"
        )
        assert fact["value"] == 0
        assert fact["status"] == "observed"
    else:
        fact = next(
            f
            for f in doc.facts
            if f["metric"] == "average_time_to_refixing"
            and f["dimensions"].get("debt_class") == "Foreign currency debt"
            and f["dimensions"].get("aggregation") == "monthly_mean"
        )
        assert fact["value"] == -0.9


def test_pdf_format_change_or_extractor_failure_is_explicit():
    with pytest.raises(ValueError):
        parse_monthly_report(b"not a PDF", spec=MONTHLY_REPORTS_2026[-1])
    with (
        patch("dalio.data_sources.riksgalden_debt.subprocess.run", side_effect=FileNotFoundError),
        pytest.raises(ValueError, match="pdftotext"),
    ):
        parse_monthly_report(b"%PDF-1.7", spec=MONTHLY_REPORTS_2026[-1])


def test_duplicate_securities_and_missing_observed_cells_fail_closed():
    text = (FIXTURES / "riksgalden_aug_2026_layout.txt").read_text()
    row = next(line for line in text.splitlines() if line.startswith("SGB 1059"))
    with pytest.raises(ValueError, match="Duplicated security"):
        monthly(text.replace(row, row + "\n" + row, 1))
    cells = json.loads((FIXTURES / "riksgalden_2026_1_cells.json").read_text())
    del cells["F17"]["C33"]
    with pytest.raises(ValueError, match="Missing reported observation"):
        parse_funding_workbook(workbook(cells))
