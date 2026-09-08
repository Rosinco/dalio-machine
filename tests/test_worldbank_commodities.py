"""World Bank Pink Sheet adapter and immutable release-pipeline tests."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from datetime import UTC, date, datetime
from pathlib import Path
from unittest.mock import MagicMock
from xml.sax.saxutils import escape

import pandas as pd
import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from dalio.data_sources.worldbank_commodities import (
    CATALOGUE_GENERATOR_VERSION,
    CATALOGUE_SCHEMA_VERSION,
    PINK_SHEET_LANDING_URL,
    PINK_SHEET_MONTHLY_URL,
    SOURCE_WORLD_BANK_COMMODITIES,
    WORLD_CODE,
    CommodityDataset,
    CommoditySeries,
    PinkSheetSource,
    canonical_series_id,
    parse_pink_sheet_workbook,
    pink_sheet_vintage_label,
)
from dalio.pipelines import fetch_commodities
from dalio.pipelines.fetch_commodities import partition_key_for, run_pipeline
from dalio.storage.db import DataRelease, Observation, ReleaseObservation, make_engine
from dalio.storage.releases import load_vintage_panel, release_history

_LONG_COLUMNS = ["country", "indicator", "date", "value", "source", "series_id"]


def _xlsx_bytes(
    series: list[tuple[str, str]],
    rows: list[tuple[str, list[float | str | None]]],
    *,
    sheet_name: str = "Monthly Prices",
    index_series: tuple[str, ...] = ("Total Index", "Energy", "Other Raw Mat."),
    index_base_year: int = 2010,
    index_title: str | None = None,
) -> bytes:
    """Build the small standards-compliant XLSX needed by these offline tests."""

    strings: list[str] = ["Monthly Prices (Nominal US dollars)", "Date"]
    strings.extend(name for name, _unit in series)
    strings.append("Unit")
    strings.extend(unit for _name, unit in series)
    strings.extend(period for period, _values in rows)
    index_title = index_title or (
        "monthly indices based on nominal US dollars, "
        f"{index_base_year}=100, 1960 to present"
    )
    strings.extend([index_title, *index_series])
    positions = {value: index for index, value in enumerate(strings)}

    def text_cell(reference: str, value: str) -> str:
        return f'<c r="{reference}" t="s"><v>{positions[value]}</v></c>'

    def column_name(position: int) -> str:
        value = position
        result = ""
        while value:
            value, remainder = divmod(value - 1, 26)
            result = chr(65 + remainder) + result
        return result

    xml_rows = [f'<row r="1">{text_cell("A1", strings[0])}</row>']
    header_cells = [text_cell("A5", "Date")]
    unit_cells = [text_cell("A6", "Unit")]
    for column, (name, unit) in enumerate(series, start=2):
        letter = column_name(column)
        header_cells.append(text_cell(f"{letter}5", name))
        unit_cells.append(text_cell(f"{letter}6", unit))
    xml_rows.append(f'<row r="5">{"".join(header_cells)}</row>')
    xml_rows.append(f'<row r="6">{"".join(unit_cells)}</row>')
    for row_number, (period, values) in enumerate(rows, start=7):
        cells = [text_cell(f"A{row_number}", period)]
        for column, value in enumerate(values, start=2):
            if value is not None:
                letter = column_name(column)
                if isinstance(value, str) and value.startswith("#"):
                    cells.append(f'<c r="{letter}{row_number}" t="e"><v>{value}</v></c>')
                elif isinstance(value, str):
                    if value not in positions:
                        positions[value] = len(strings)
                        strings.append(value)
                    cells.append(text_cell(f"{letter}{row_number}", value))
                else:
                    cells.append(f'<c r="{letter}{row_number}"><v>{value}</v></c>')
        xml_rows.append(f'<row r="{row_number}">{"".join(cells)}</row>')

    shared = "".join(f"<si><t>{escape(value)}</t></si>" for value in strings)
    workbook = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f'<sheets><sheet name="{escape(sheet_name)}" sheetId="1" r:id="rId1"/>'
        '<sheet name="Monthly Indices" sheetId="2" r:id="rId2"/></sheets>'
        "</workbook>"
    )
    worksheet = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<sheetData>{"".join(xml_rows)}</sheetData></worksheet>'
    )
    index_xml_rows = [
        f'<row r="1">{text_cell("A1", strings[0])}</row>',
        f'<row r="2">{text_cell("A2", index_title)}</row>',
    ]
    index_headers = [
        text_cell(f"{column_name(column)}6", name)
        for column, name in enumerate(index_series, start=2)
    ]
    index_xml_rows.append(f'<row r="6">{"".join(index_headers)}</row>')
    for row_number, (period, _values) in enumerate(rows, start=10):
        cells = [text_cell(f"A{row_number}", period)]
        cells.extend(
            f'<c r="{column_name(column)}{row_number}"><v>{100 + column + row_number}</v></c>'
            for column, _name in enumerate(index_series, start=2)
        )
        index_xml_rows.append(f'<row r="{row_number}">{"".join(cells)}</row>')
    index_worksheet = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<sheetData>{"".join(index_xml_rows)}</sheetData></worksheet>'
    )
    relationships = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        'Target="worksheets/sheet1.xml"/>'
        '<Relationship Id="rId2" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        'Target="worksheets/sheet2.xml"/></Relationships>'
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/worksheets/sheet1.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        '<Override PartName="/xl/worksheets/sheet2.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        '<Override PartName="/xl/sharedStrings.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml"/>'
        "</Types>"
    )
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("xl/workbook.xml", workbook)
        archive.writestr("xl/_rels/workbook.xml.rels", relationships)
        archive.writestr("xl/sharedStrings.xml", f"<sst>{shared}</sst>")
        archive.writestr("xl/worksheets/sheet1.xml", worksheet)
        archive.writestr("xl/worksheets/sheet2.xml", index_worksheet)
    return output.getvalue()


def _fixture_workbook() -> bytes:
    return _xlsx_bytes(
        [
            ("Crude oil, Brent", "$/bbl"),
            ("Copper", "$/mt"),
            ("Tea, Colombo", "$/kg"),
        ],
        [
            ("1960M01", [1.63, 715.4, None]),
            ("1960M02", [1.63, 728.8, 0.93]),
            ("1960M03", [1.63, None, 0.94]),
        ],
    )


def _response(content: bytes, status_code: int = 200):
    response = MagicMock()
    response.content = content
    response.status_code = status_code
    response.raise_for_status.side_effect = None
    return response


def test_parser_emits_every_workbook_series_in_long_format_with_catalogue():
    parsed = parse_pink_sheet_workbook(_fixture_workbook())

    assert list(parsed.observations.columns) == _LONG_COLUMNS
    assert set(parsed.observations["country"]) == {WORLD_CODE}
    assert set(parsed.observations["series_id"]) == {
        canonical_series_id("Crude oil, Brent", "Monthly Prices"),
        canonical_series_id("Copper", "Monthly Prices"),
        canonical_series_id("Tea, Colombo", "Monthly Prices"),
        canonical_series_id("Total Index", "Monthly Indices"),
        canonical_series_id("Energy", "Monthly Indices"),
        canonical_series_id("Other Raw Mat.", "Monthly Indices"),
    }
    assert len(parsed.observations) == 16  # blank workbook cells are missing, never zero-filled
    assert parsed.observations["date"].min() == date(1960, 1, 1)
    assert parsed.observations["date"].max() == date(1960, 3, 1)

    by_benchmark = {series.benchmark: series for series in parsed.catalogue}
    assert len(by_benchmark) == 6
    assert by_benchmark["Crude oil, Brent"].unit == "$/bbl"
    assert by_benchmark["Crude oil, Brent"].series_id == "monthly_prices:crude_oil_brent"
    assert by_benchmark["Crude oil, Brent"].category == "energy"
    assert by_benchmark["Crude oil, Brent"].curated is True
    assert by_benchmark["Copper"].category == "base_metals"
    assert by_benchmark["Copper"].curated is True
    assert by_benchmark["Tea, Colombo"].category == "food_and_beverages"
    assert by_benchmark["Tea, Colombo"].curated is False
    assert by_benchmark["Total Index"].indicator == "commodity_index_total_index"
    assert by_benchmark["Total Index"].series_id == "monthly_indices:total_index"
    assert by_benchmark["Total Index"].unit == "2010=100"
    assert by_benchmark["Total Index"].currency is None
    assert by_benchmark["Total Index"].price_basis == "nominal_usd_index_2010_100"
    assert by_benchmark["Total Index"].curated is True
    assert by_benchmark["Energy"].category == "energy"
    assert by_benchmark["Other Raw Mat."].category == "agricultural_raw_materials"
    assert {series.worksheet for series in parsed.catalogue} == {
        "Monthly Prices",
        "Monthly Indices",
    }
    assert parsed.workbook_sha256 == hashlib.sha256(_fixture_workbook()).hexdigest()


@pytest.mark.parametrize(
    "period",
    ["1960", "1960M00", "1960M13", "January 1960", "not-a-date"],
)
def test_parser_fails_closed_on_non_monthly_periods(period):
    workbook = _xlsx_bytes([("Copper", "$/mt")], [(period, [700.0])])
    with pytest.raises(ValueError, match="monthly observation rows"):
        parse_pink_sheet_workbook(workbook)


def test_duplicate_native_benchmarks_fail_closed():
    workbook = _xlsx_bytes(
        [("Copper", "$/mt"), ("Copper", "$/mt")],
        [("1960M01", [700.0, 701.0])],
    )
    with pytest.raises(ValueError, match="duplicate benchmark"):
        parse_pink_sheet_workbook(workbook)


def test_unicode_ellipsis_is_a_publisher_missing_marker_not_a_value():
    workbook = _xlsx_bytes(
        [("Crude oil, WTI", "$/bbl")],
        [("1960M01", ["…"]), ("1960M02", [2.97])],
    )

    parsed = parse_pink_sheet_workbook(workbook)

    wti = parsed.observations.loc[
        parsed.observations["series_id"]
        == canonical_series_id("Crude oil, WTI", "Monthly Prices"),
        ["date", "value"],
    ]
    assert wti.to_dict("records") == [
        {"date": date(1960, 2, 1), "value": 2.97}
    ]


def test_excel_error_cell_is_a_publisher_missing_observation_not_a_value():
    workbook = _xlsx_bytes(
        [("Rice, Thai 25%", "$/mt")],
        [("1988M09", ["#VALUE!"]), ("1988M10", [250.0])],
    )

    parsed = parse_pink_sheet_workbook(workbook)

    rice = parsed.observations.loc[
        parsed.observations["series_id"]
        == canonical_series_id("Rice, Thai 25%", "Monthly Prices"),
        ["date", "value"],
    ]
    assert rice.to_dict("records") == [
        {"date": date(1988, 10, 1), "value": 250.0}
    ]


def test_nonpositive_cells_are_quarantined_including_known_rice_placeholders():
    workbook = _xlsx_bytes(
        [("Rice, Thai 25%", "$/mt")],
        [
            ("2008M01", [364.4]),
            ("2008M02", ["#VALUE!"]),
            ("2008M03", [0.0]),
            ("2008M04", [0]),
            ("2008M05", [-1.0]),
            ("2008M06", [0.0]),
            ("2008M07", [700.0]),
        ],
    )

    parsed = parse_pink_sheet_workbook(workbook)

    rice = parsed.observations.loc[
        parsed.observations["indicator"] == "commodity_price_rice_thai_25",
        ["date", "value"],
    ]
    assert rice.to_dict("records") == [
        {"date": date(2008, 1, 1), "value": 364.4},
        {"date": date(2008, 7, 1), "value": 700.0},
    ]
    assert (parsed.observations["value"] > 0).all()
    assert [
        {
            "worksheet": issue.worksheet,
            "cell": issue.cell,
            "date": issue.date,
            "benchmark": issue.benchmark,
            "raw_value": issue.raw_value,
            "reason": issue.reason,
        }
        for issue in parsed.quality_issues
    ] == [
        {
            "worksheet": "Monthly Prices",
            "cell": "B9",
            "date": date(2008, 3, 1),
            "benchmark": "Rice, Thai 25%",
            "raw_value": 0.0,
            "reason": "nonpositive_value",
        },
        {
            "worksheet": "Monthly Prices",
            "cell": "B10",
            "date": date(2008, 4, 1),
            "benchmark": "Rice, Thai 25%",
            "raw_value": 0.0,
            "reason": "nonpositive_value",
        },
        {
            "worksheet": "Monthly Prices",
            "cell": "B11",
            "date": date(2008, 5, 1),
            "benchmark": "Rice, Thai 25%",
            "raw_value": -1.0,
            "reason": "nonpositive_value",
        },
        {
            "worksheet": "Monthly Prices",
            "cell": "B12",
            "date": date(2008, 6, 1),
            "benchmark": "Rice, Thai 25%",
            "raw_value": 0.0,
            "reason": "nonpositive_value",
        },
    ]


def test_display_label_aliases_share_stable_series_id_and_indicator():
    old = parse_pink_sheet_workbook(
        _xlsx_bytes([("Coal, Australia **", "$/mt")], [("1960M01", [20.0])])
    )
    current = parse_pink_sheet_workbook(
        _xlsx_bytes([("Coal, Australian", "$/mt")], [("1960M01", [20.0])])
    )

    old_series = next(series for series in old.catalogue if series.series_kind == "price")
    current_series = next(
        series for series in current.catalogue if series.series_kind == "price"
    )
    assert old_series.benchmark == "Coal, Australia **"
    assert current_series.benchmark == "Coal, Australian"
    assert old_series.series_id == current_series.series_id == (
        "monthly_prices:coal_australian"
    )
    assert old_series.indicator == current_series.indicator == (
        "commodity_price_coal_australian"
    )


def test_index_semantics_are_derived_from_each_publisher_unit_and_rebase():
    parsed = parse_pink_sheet_workbook(
        _xlsx_bytes(
            [("Natural gas index", "(2020=100)"), ("Cotton, A Index", "$/kg")],
            [("1960M01", [12.0, 1.5])],
            index_base_year=2020,
        )
    )
    by_benchmark = {series.benchmark: series for series in parsed.catalogue}

    gas = by_benchmark["Natural gas index"]
    assert gas.series_kind == "index"
    assert gas.indicator == "commodity_index_natural_gas_index"
    assert gas.unit == "2020=100"
    assert gas.currency is None
    assert gas.price_basis == "nominal_usd_index_2020_100"
    assert by_benchmark["Cotton, A Index"].series_kind == "price"
    assert by_benchmark["Total Index"].unit == "2020=100"
    assert by_benchmark["Total Index"].price_basis == "nominal_usd_index_2020_100"


def test_index_sheet_without_an_unambiguous_base_fails_closed():
    workbook = _xlsx_bytes(
        [("Copper", "$/mt")],
        [("1960M01", [700.0])],
        index_title="monthly indices in nominal US dollar terms",
    )

    with pytest.raises(ValueError, match="index base"):
        parse_pink_sheet_workbook(workbook)


def test_monthly_sheet_periods_must_be_contiguous():
    workbook = _xlsx_bytes(
        [("Copper", "$/mt")],
        [("1960M01", [700.0]), ("1960M03", [710.0])],
    )

    with pytest.raises(ValueError, match="contiguous"):
        parse_pink_sheet_workbook(workbook)


def test_fetch_validates_then_caches_and_archives_content_addressed_bytes(tmp_path):
    workbook = _fixture_workbook()
    client = MagicMock()
    client.get.return_value = _response(workbook)
    cache_dir = tmp_path / "cache"
    artifact_dir = tmp_path / "artifacts"
    source = PinkSheetSource(
        client=client,
        cache_dir=cache_dir,
        artifact_dir=artifact_dir,
        retry_backoff_seconds=0,
        source_url=PINK_SHEET_MONTHLY_URL,
    )

    first = source.fetch(use_cache=True)
    second = source.fetch(use_cache=True)

    assert client.get.call_count == 1
    assert client.get.call_args.args[0] == PINK_SHEET_MONTHLY_URL
    assert first.workbook_sha256 == second.workbook_sha256
    assert first.artifact_path is not None
    assert first.artifact_path.name == f"{first.workbook_sha256}.xlsx"
    assert first.artifact_path.read_bytes() == workbook
    assert first.catalogue_path is not None
    assert first.catalogue_path.name == (
        f"{first.workbook_sha256}.catalogue-v{CATALOGUE_SCHEMA_VERSION}-"
        f"{CATALOGUE_GENERATOR_VERSION}.json"
    )
    catalogue = json.loads(first.catalogue_path.read_text())
    assert catalogue["schema_version"] == CATALOGUE_SCHEMA_VERSION
    assert catalogue["generator_version"] == CATALOGUE_GENERATOR_VERSION
    assert catalogue["workbook_sha256"] == first.workbook_sha256
    assert catalogue["source"] == SOURCE_WORLD_BANK_COMMODITIES
    assert catalogue["quality"] == {
        "quarantined_cell_count": 0,
        "quarantined_cells": [],
    }
    assert len(catalogue["series"]) == 6
    assert set(catalogue["series"][0]) == {
        "benchmark",
        "category",
        "currency",
        "curated",
        "frequency",
        "indicator",
        "price_basis",
        "series_id",
        "series_kind",
        "unit",
        "worksheet",
    }
    assert len(list(cache_dir.glob("*.xlsx"))) == 1
    assert len(list(artifact_dir.rglob("*.xlsx"))) == 1
    assert len(list(artifact_dir.rglob("*.catalogue-*.json"))) == 1
    pd.testing.assert_frame_equal(first.observations, second.observations)


def test_versioned_catalogue_coexists_with_an_older_generator_for_same_workbook(tmp_path):
    workbook = _fixture_workbook()
    digest = hashlib.sha256(workbook).hexdigest()
    artifact_dir = tmp_path / "artifacts"
    legacy = artifact_dir / digest[:2] / f"{digest}.catalogue.json"
    legacy.parent.mkdir(parents=True)
    legacy.write_text('{"schema_version":1}\n')
    client = MagicMock()
    client.get.return_value = _response(workbook)
    source = PinkSheetSource(
        client=client,
        cache_dir=tmp_path / "cache",
        artifact_dir=artifact_dir,
        source_url=PINK_SHEET_MONTHLY_URL,
    )

    dataset = source.fetch(use_cache=False)

    assert legacy.read_text() == '{"schema_version":1}\n'
    assert dataset.catalogue_path is not None
    assert dataset.catalogue_path != legacy
    assert dataset.catalogue_path.exists()


def test_catalogue_persists_quarantined_cell_evidence(tmp_path):
    workbook = _xlsx_bytes(
        [("Rice, Thai 25%", "$/mt")],
        [("2008M03", [0.0]), ("2008M04", [450.0])],
    )
    client = MagicMock()
    client.get.return_value = _response(workbook)
    source = PinkSheetSource(
        client=client,
        cache_dir=tmp_path / "cache",
        artifact_dir=tmp_path / "artifacts",
        source_url=PINK_SHEET_MONTHLY_URL,
    )

    dataset = source.fetch(use_cache=False)

    assert dataset.catalogue_path is not None
    catalogue = json.loads(dataset.catalogue_path.read_text())
    assert catalogue["quality"] == {
        "quarantined_cell_count": 1,
        "quarantined_cells": [
            {
                "benchmark": "Rice, Thai 25%",
                "cell": "B7",
                "date": "2008-03-01",
                "raw_value": 0.0,
                "reason": "nonpositive_value",
                "worksheet": "Monthly Prices",
            }
        ],
    }


def test_invalid_download_never_poison_caches_or_artifact_archive(tmp_path):
    client = MagicMock()
    client.get.return_value = _response(b"this is not an xlsx")
    cache_dir = tmp_path / "cache"
    artifact_dir = tmp_path / "artifacts"
    source = PinkSheetSource(
        client=client,
        cache_dir=cache_dir,
        artifact_dir=artifact_dir,
        attempts=1,
        source_url=PINK_SHEET_MONTHLY_URL,
    )

    with pytest.raises(ValueError, match="valid XLSX"):
        source.fetch(use_cache=True)

    assert list(cache_dir.rglob("*.*")) == []
    assert list(artifact_dir.rglob("*.*")) == []


def test_permanent_http_error_is_not_retried(tmp_path):
    client = MagicMock()
    client.get.return_value = _response(b"missing", status_code=404)
    source = PinkSheetSource(
        client=client,
        cache_dir=tmp_path / "cache",
        artifact_dir=tmp_path / "artifacts",
        attempts=3,
        retry_backoff_seconds=0,
        source_url=PINK_SHEET_MONTHLY_URL,
    )

    with pytest.raises(ValueError, match="404"):
        source.fetch(use_cache=False)

    assert client.get.call_count == 1


def test_default_source_discovers_the_current_monthly_workbook_on_official_hosts(tmp_path):
    workbook = _fixture_workbook()
    current_url = (
        "https://thedocs.worldbank.org/en/doc/new-release/related/"
        "CMO-Historical-Data-Monthly.xlsx"
    )
    landing = MagicMock()
    landing.status_code = 200
    landing.content = (
        f'<html><a href="{current_url}">Monthly historical data</a></html>'.encode()
    )
    landing.raise_for_status.side_effect = None
    client = MagicMock()
    client.get.side_effect = [landing, _response(workbook)]
    source = PinkSheetSource(
        client=client,
        cache_dir=tmp_path / "cache",
        artifact_dir=tmp_path / "artifacts",
        retry_backoff_seconds=0,
    )

    dataset = source.fetch(use_cache=False)

    assert [call.args[0] for call in client.get.call_args_list] == [
        PINK_SHEET_LANDING_URL,
        current_url,
    ]
    assert dataset.source_url == current_url


@pytest.mark.parametrize(
    "href",
    [
        "http://thedocs.worldbank.org/CMO-Historical-Data-Monthly.xlsx",
        "https://worldbank.example/CMO-Historical-Data-Monthly.xlsx",
        "https://evil.example/?file=CMO-Historical-Data-Monthly.xlsx",
    ],
)
def test_discovery_rejects_non_https_or_non_world_bank_workbook_links(tmp_path, href):
    landing = MagicMock()
    landing.status_code = 200
    landing.content = f'<html><a href="{href}">bad</a></html>'.encode()
    landing.raise_for_status.side_effect = None
    client = MagicMock()
    client.get.return_value = landing
    source = PinkSheetSource(
        client=client,
        cache_dir=tmp_path / "cache",
        artifact_dir=tmp_path / "artifacts",
        attempts=1,
    )

    with pytest.raises(ValueError, match="discover"):
        source.fetch(use_cache=False)

    assert client.get.call_count == 1


def _series(
    benchmark: str,
    indicator: str,
    unit: str,
    category: str,
    curated: bool,
    *,
    worksheet: str = "Monthly Prices",
    series_kind: str = "price",
) -> CommoditySeries:
    return CommoditySeries(
        indicator=indicator,
        series_id=canonical_series_id(benchmark, worksheet),
        benchmark=benchmark,
        unit=unit,
        currency=None if series_kind == "index" else "USD",
        frequency="monthly",
        price_basis=(
            "nominal_usd_index_2010_100"
            if series_kind == "index"
            else "nominal_monthly_average"
        ),
        category=category,
        curated=curated,
        worksheet=worksheet,
        series_kind=series_kind,
    )


def _dataset(
    values: dict[CommoditySeries, list[tuple[date, float]]],
    artifact_path: Path,
    sha256: str = "a" * 64,
) -> CommodityDataset:
    observations = pd.DataFrame(
        [
            {
                "country": WORLD_CODE,
                "indicator": series.indicator,
                "date": observed_on,
                "value": value,
                "source": SOURCE_WORLD_BANK_COMMODITIES,
                "series_id": series.series_id,
            }
            for series, series_values in values.items()
            for observed_on, value in series_values
        ],
        columns=_LONG_COLUMNS,
    )
    return CommodityDataset(
        observations=observations,
        catalogue=tuple(values),
        workbook_sha256=sha256,
        source_url=PINK_SHEET_MONTHLY_URL,
        artifact_path=artifact_path,
    )


class _FakeSource:
    def __init__(self, dataset: CommodityDataset):
        self.dataset = dataset
        self.calls: list[bool] = []

    def fetch(self, *, use_cache=True):
        self.calls.append(use_cache)
        return self.dataset


def _at(month: int) -> datetime:
    return datetime(2026, month, 8, 12, tzinfo=UTC)


def test_pipeline_stores_one_complete_wld_native_series_release_per_commodity(tmp_path):
    brent = _series(
        "Crude oil, Brent",
        "commodity_price_crude_oil_brent",
        "$/bbl",
        "energy",
        True,
    )
    tea = _series(
        "Tea, Colombo",
        "commodity_price_tea_colombo",
        "$/kg",
        "food_and_beverages",
        False,
    )
    artifact = tmp_path / "artifacts" / f"{'a' * 64}.xlsx"
    source = _FakeSource(
        _dataset(
            {
                brent: [(date(1960, 1, 1), 1.63), (date(1960, 2, 1), 1.64)],
                tea: [(date(1960, 2, 1), 0.93)],
            },
            artifact,
        )
    )
    engine = make_engine(tmp_path / "commodities.db")

    summary = run_pipeline(
        source=source,
        use_cache=False,
        engine=engine,
        retrieved_at=_at(9),
        allow_contraction=True,
    )

    assert source.calls == [False]
    assert summary["series_total"] == 2
    assert summary["curated_series"] == 1
    assert summary["rows"] == 3
    assert summary["created_releases"] == 2
    assert summary["inserted"] == 3
    assert summary["workbook_sha256"] == "a" * 64
    assert summary["artifact_path"] == str(artifact)
    with Session(engine) as session:
        releases = session.scalars(select(DataRelease).order_by(DataRelease.partition_key)).all()
        current = session.execute(
            select(
                Observation.country,
                Observation.indicator,
                Observation.date,
                Observation.value,
            ).order_by(Observation.indicator, Observation.date)
        ).all()
        ledger_count = session.scalar(select(func.count()).select_from(ReleaseObservation))

    assert {release.partition_key for release in releases} == {
        partition_key_for(brent),
        partition_key_for(tea),
    }
    assert {release.source_family for release in releases} == {
        SOURCE_WORLD_BANK_COMMODITIES
    }
    assert {release.source_url for release in releases} == {PINK_SHEET_MONTHLY_URL}
    assert all(
        release.vintage_label == pink_sheet_vintage_label("a" * 64)
        for release in releases
    )
    assert all(
        f"catalogue:v{CATALOGUE_SCHEMA_VERSION}/{CATALOGUE_GENERATOR_VERSION}"
        in release.vintage_label
        for release in releases
    )
    assert current == [
        (WORLD_CODE, brent.indicator, date(1960, 1, 1), 1.63),
        (WORLD_CODE, brent.indicator, date(1960, 2, 1), 1.64),
        (WORLD_CODE, tea.indicator, date(1960, 2, 1), 0.93),
    ]
    assert ledger_count == 3


def test_pipeline_revision_records_the_complete_workbook_vintage_and_projects_changes(tmp_path):
    brent = _series(
        "Crude oil, Brent",
        "commodity_price_crude_oil_brent",
        "$/bbl",
        "energy",
        True,
    )
    tea = _series(
        "Tea, Colombo",
        "commodity_price_tea_colombo",
        "$/kg",
        "food_and_beverages",
        False,
    )
    source = _FakeSource(
        _dataset(
            {
                brent: [(date(2026, 6, 1), 75.0), (date(2026, 7, 1), 76.0)],
                tea: [(date(2026, 7, 1), 3.0)],
            },
            tmp_path / "a.xlsx",
        )
    )
    engine = make_engine(tmp_path / "commodities.db")
    run_pipeline(
        source=source,
        engine=engine,
        retrieved_at=_at(8),
        allow_contraction=True,
    )

    source.dataset = _dataset(
        {
            brent: [(date(2026, 6, 1), 74.0)],
            tea: [(date(2026, 7, 1), 3.0)],
        },
        tmp_path / "b.xlsx",
        sha256="b" * 64,
    )
    second = run_pipeline(
        source=source,
        engine=engine,
        retrieved_at=_at(9),
        allow_contraction=True,
    )

    with Session(engine) as session:
        old = load_vintage_panel(
            session,
            _at(8),
            partition_keys=(partition_key_for(brent), partition_key_for(tea)),
        )
        new = load_vintage_panel(
            session,
            _at(9),
            partition_keys=(partition_key_for(brent), partition_key_for(tea)),
        )
        brent_history = release_history(session, partition_key_for(brent))
        tea_history = release_history(session, partition_key_for(tea))

    # A new official workbook SHA is a new semantic vintage for every native
    # partition, even when a sibling's numeric rows happen to be unchanged.
    assert second["created_releases"] == 2
    assert second["removed"] == 1
    assert len(brent_history) == 2
    assert len(tea_history) == 2
    assert old.loc[old["indicator"] == brent.indicator, "value"].tolist() == [75.0, 76.0]
    assert new.loc[new["indicator"] == brent.indicator, "value"].tolist() == [74.0]
    assert new.loc[new["indicator"] == tea.indicator, "value"].tolist() == [3.0]


def test_pipeline_identical_workbook_poll_is_idempotent(tmp_path):
    copper = _series(
        "Copper", "commodity_price_copper", "$/mt", "base_metals", True
    )
    source = _FakeSource(
        _dataset(
            {copper: [(date(1960, 1, 1), 700.0)]},
            tmp_path / "same.xlsx",
        )
    )
    engine = make_engine(tmp_path / "commodities.db")

    first = run_pipeline(
        source=source,
        engine=engine,
        retrieved_at=_at(8),
        allow_contraction=True,
    )
    second = run_pipeline(
        source=source,
        engine=engine,
        retrieved_at=_at(9),
        allow_contraction=True,
    )

    assert first["created_releases"] == 1
    assert second["created_releases"] == 0
    assert second["inserted"] == 0
    assert second["skipped"] == 1
    with Session(engine) as session:
        assert len(release_history(session, partition_key_for(copper))) == 1


def test_pipeline_fails_closed_when_catalogue_and_observations_disagree(tmp_path):
    copper = _series(
        "Copper", "commodity_price_copper", "$/mt", "base_metals", True
    )
    dataset = _dataset(
        {copper: [(date(2026, 7, 1), 9_000.0)]},
        tmp_path / "artifact.xlsx",
    )
    dataset = CommodityDataset(
        observations=dataset.observations.assign(series_id="unexpected"),
        catalogue=dataset.catalogue,
        workbook_sha256=dataset.workbook_sha256,
        source_url=dataset.source_url,
        artifact_path=dataset.artifact_path,
    )
    engine = make_engine(tmp_path / "commodities.db")

    result = run_pipeline(
        source=_FakeSource(dataset),
        engine=engine,
        retrieved_at=_at(9),
        allow_contraction=True,
    )

    assert "catalogue" in result["error"]
    with Session(engine) as session:
        assert session.scalar(select(func.count()).select_from(DataRelease)) == 0
        assert session.scalar(select(func.count()).select_from(Observation)) == 0


def test_pipeline_label_drift_reuses_one_canonical_partition(tmp_path):
    old = parse_pink_sheet_workbook(
        _xlsx_bytes([("Coal, Australia **", "$/mt")], [("1960M01", [20.0])])
    )
    current = parse_pink_sheet_workbook(
        _xlsx_bytes([("Coal, Australian", "$/mt")], [("1960M01", [20.0])])
    )
    engine = make_engine(tmp_path / "commodities.db")

    first = run_pipeline(
        source=_FakeSource(old),
        engine=engine,
        retrieved_at=_at(8),
        allow_contraction=True,
    )
    second = run_pipeline(
        source=_FakeSource(current),
        engine=engine,
        retrieved_at=_at(9),
        allow_contraction=True,
    )
    coal = next(
        series for series in current.catalogue if series.benchmark == "Coal, Australian"
    )

    assert "error" not in first
    assert "error" not in second
    with Session(engine) as session:
        history = release_history(session, partition_key_for(coal))
        panel = load_vintage_panel(
            session,
            _at(9),
            indicators=(coal.indicator,),
        )
    assert len(history) == 2
    assert panel[["series_id", "value"]].to_dict("records") == [
        {"series_id": "monthly_prices:coal_australian", "value": 20.0}
    ]


def test_pipeline_rejects_an_initial_truncated_catalogue_without_override(tmp_path):
    copper = _series(
        "Copper", "commodity_price_copper", "$/mt", "base_metals", True
    )
    dataset = _dataset(
        {copper: [(date(1960, 1, 1), 700.0)]},
        tmp_path / "artifact.xlsx",
    )
    engine = make_engine(tmp_path / "commodities.db")

    result = run_pipeline(
        source=_FakeSource(dataset),
        engine=engine,
        retrieved_at=_at(9),
    )

    assert "catalogue" in result["error"]
    with Session(engine) as session:
        assert session.scalar(select(func.count()).select_from(DataRelease)) == 0


def test_pipeline_rejects_catalogue_substitution_even_when_column_count_is_unchanged(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(fetch_commodities, "EXPECTED_MIN_PRICE_COLUMNS", 1)
    monkeypatch.setattr(fetch_commodities, "EXPECTED_MIN_INDEX_COLUMNS", 0)
    monkeypatch.setattr(fetch_commodities, "MIN_OBSERVATIONS_PER_SERIES", 1)
    monkeypatch.setattr(fetch_commodities, "MIN_HISTORY_MONTHS", 1)
    monkeypatch.setattr(fetch_commodities, "MAX_LATEST_LAG_MONTHS", 1_000)
    monkeypatch.setattr(
        fetch_commodities,
        "EXPECTED_SERIES_IDS_BY_WORKSHEET",
        {
            "Monthly Prices": frozenset(
                {canonical_series_id("Copper", "Monthly Prices")}
            ),
            "Monthly Indices": frozenset(),
        },
    )
    zinc = _series("Zinc", "commodity_price_zinc", "$/mt", "base_metals", True)
    dataset = _dataset(
        {zinc: [(date(1960, 1, 1), 1_000.0)]},
        tmp_path / "substituted.xlsx",
    )
    engine = make_engine(tmp_path / "commodities.db")

    result = run_pipeline(
        source=_FakeSource(dataset),
        engine=engine,
        retrieved_at=_at(9),
    )

    assert "missing expected series" in result["error"]
    with Session(engine) as session:
        assert session.scalar(select(func.count()).select_from(DataRelease)) == 0


def test_pipeline_blocks_history_contraction_until_explicit_override(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(fetch_commodities, "EXPECTED_MIN_PRICE_COLUMNS", 1)
    monkeypatch.setattr(fetch_commodities, "EXPECTED_MIN_INDEX_COLUMNS", 0)
    monkeypatch.setattr(fetch_commodities, "MIN_OBSERVATIONS_PER_SERIES", 1)
    monkeypatch.setattr(fetch_commodities, "MIN_HISTORY_MONTHS", 1)
    monkeypatch.setattr(fetch_commodities, "MAX_LATEST_LAG_MONTHS", 1_000)
    monkeypatch.setattr(
        fetch_commodities,
        "EXPECTED_SERIES_IDS_BY_WORKSHEET",
        {
            "Monthly Prices": frozenset(
                {canonical_series_id("Copper", "Monthly Prices")}
            ),
            "Monthly Indices": frozenset(),
        },
    )
    copper = _series(
        "Copper", "commodity_price_copper", "$/mt", "base_metals", True
    )
    engine = make_engine(tmp_path / "commodities.db")
    source = _FakeSource(
        _dataset(
            {
                copper: [
                    (date(1960, 1, 1), 700.0),
                    (date(1960, 2, 1), 710.0),
                ]
            },
            tmp_path / "first.xlsx",
        )
    )
    seed = run_pipeline(
        source=source,
        engine=engine,
        retrieved_at=_at(8),
        allow_contraction=True,
    )
    assert "error" not in seed
    source.dataset = _dataset(
        {copper: [(date(1960, 1, 1), 705.0)]},
        tmp_path / "second.xlsx",
        sha256="b" * 64,
    )

    blocked = run_pipeline(source=source, engine=engine, retrieved_at=_at(9))

    assert "contracts" in blocked["error"]
    with Session(engine) as session:
        assert session.scalars(
            select(Observation.value).where(Observation.indicator == copper.indicator)
        ).all() == [700.0, 710.0]

    allowed = run_pipeline(
        source=source,
        engine=engine,
        retrieved_at=_at(9),
        allow_contraction=True,
    )
    assert "error" not in allowed
    with Session(engine) as session:
        assert session.scalars(
            select(Observation.value).where(Observation.indicator == copper.indicator)
        ).all() == [705.0]


def test_pipeline_rolls_back_the_whole_workbook_if_one_series_fails(
    tmp_path, monkeypatch
):
    brent = _series(
        "Crude oil, Brent",
        "commodity_price_crude_oil_brent",
        "$/bbl",
        "energy",
        True,
    )
    tea = _series(
        "Tea, Colombo",
        "commodity_price_tea_colombo",
        "$/kg",
        "food_and_beverages",
        False,
    )
    engine = make_engine(tmp_path / "commodities.db")
    source = _FakeSource(
        _dataset(
            {
                brent: [(date(1960, 1, 1), 1.63)],
                tea: [(date(1960, 1, 1), 0.93)],
            },
            tmp_path / "first.xlsx",
        )
    )
    seed = run_pipeline(
        source=source,
        engine=engine,
        retrieved_at=_at(8),
        allow_contraction=True,
    )
    assert seed["created_releases"] == 2
    source.dataset = _dataset(
        {
            brent: [(date(1960, 1, 1), 2.0)],
            tea: [(date(1960, 1, 1), 1.0)],
        },
        tmp_path / "second.xlsx",
        sha256="b" * 64,
    )
    real_ingest = fetch_commodities.ingest_release_snapshot
    calls = 0

    def fail_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected second-series failure")
        return real_ingest(*args, **kwargs)

    monkeypatch.setattr(fetch_commodities, "ingest_release_snapshot", fail_second)

    failed = run_pipeline(
        source=source,
        engine=engine,
        retrieved_at=_at(9),
        allow_contraction=True,
    )

    assert failed["error"] == "injected second-series failure"
    assert failed["rolled_back"] is True
    assert failed["created_releases"] == 0
    with Session(engine) as session:
        current = session.execute(
            select(Observation.indicator, Observation.value).order_by(Observation.indicator)
        ).all()
        release_count = session.scalar(select(func.count()).select_from(DataRelease))
    assert current == [(brent.indicator, 1.63), (tea.indicator, 0.93)]
    assert release_count == 2
