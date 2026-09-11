"""Comparable presentation without erasing native Nordic definitions or gaps."""

import re
from copy import deepcopy

import pytest

from dalio.assessments.core import content_hash
from dalio.nordic_monitoring.render import render_country, render_index
from tests.test_nordic_monitoring_core import build
from tests.test_nordic_monitoring_core import inputs as inputs


def reseal(snapshot):
    snapshot["snapshot_sha256"] = content_hash({k: v for k, v in snapshot.items() if k != "snapshot_sha256"})
    return snapshot


def test_native_date_only_update_does_not_become_an_invented_timestamp(inputs):
    source = next(s for s in inputs[0]["series"] if s["country"] == "FI" and s["indicator"] == "yield_10y")
    source["publisher_updated_at"] = None
    source["publisher_metadata"].update(report_updated_date="2026-09-10", update_precision="date",
                                         update_timezone="Europe/Helsinki")
    text = render_country(build(inputs), "FI")
    assert "Report updated (date only): 2026-09-10; publisher timezone: Europe/Helsinki" in text
    assert "no exact publication or update time" in text
    assert "Dataset updated: 2026-09-10T00:00:00" not in text


def test_index_shows_fixed_country_order_explicit_windows_and_scope_differences(inputs):
    snapshot = build(inputs)
    text = render_index(snapshot)
    assert [text.index(f"]({code}.md)") for code in ("SE", "NO", "DK", "FI")] == sorted(
        text.index(f"]({code}.md)") for code in ("SE", "NO", "DK", "FI"))
    assert "Mean 2026-05–2026-07 versus 2026-02–2026-04" in text
    assert "2026-09-08 versus 2026-06-09 (90-day target 2026-06-10)" in text
    assert "percentage points" in text and "index, 2021=100" in text
    assert "Native definitions" in text and "NO original definition" in text
    assert "ranking" in text and "Swedish order intake is supplemental" in text
    assert "context/NO.md" in text and "known-at" in text and "UTC" in text


def test_country_report_contains_only_used_native_country_citations(inputs):
    snapshot = build(inputs)
    text = render_country(snapshot, "NO")
    assert "# Norway" in text and "context/NO.md" in text
    assert "[Nordic comparison](index.md)" in text
    assert "No current directional" not in text
    assert "Evidence that would challenge" in text
    assert "NO original definition" in text and "DK original definition" not in text
    anchors = set(re.findall(r'<a id="(evidence-[a-f0-9]+)"', text))
    local_targets = set(re.findall(r'\]\(#(evidence-[a-f0-9]+)\)', text))
    assert local_targets and local_targets <= anchors
    expected = set()
    for country in snapshot["countries"]:
        if country["country"] == "NO":
            for signal in country["signals"]:
                expected.update(signal["evidence_refs"])
                for link in signal["scenario_links"]:
                    expected.update(link["reference_evidence_refs"])
    assert len(anchors) == len(expected)
    for ref in expected:
        assert snapshot["citations"][ref]["country"] == "NO"


def test_missing_latest_and_stale_rates_do_not_disappear_from_comparison(inputs):
    norway = inputs[0]["series"]
    norway[0]["observations"][-1].update(value=None, status="not_reported")
    norway[2]["observations"].pop()
    snapshot = build(inputs)
    for text in (render_index(snapshot), render_country(snapshot, "NO")):
        assert "Missing latest" in text and "Stale" in text
        assert "Not available" in text and "2026-07" in text
    country = render_country(snapshot, "NO")
    assert "no older-value backfill" in country
    assert "No current directional reading" in country


def test_monthly_policy_instrument_keeps_its_own_frequency_and_window(inputs):
    policy = next(s for s in inputs[0]["series"] if s['country'] == 'DK' and s['indicator'] == 'policy_rate')
    loan = next(s for s in inputs[0]["series"] if s['country'] == 'DK' and s['indicator'] == 'corporate_new_lending_rate')
    policy.update(frequency='monthly', observations=deepcopy(loan['observations']),
                  definition='Monthly average certificate-of-deposit rate.')
    policy['spec'].update(frequency='monthly', comparison='three_month_rate',
                          label='Certificates of deposit: monthly average')
    snapshot = build(inputs)
    text = render_country(snapshot, 'DK')
    assert 'Certificates of deposit: monthly average' in text
    assert 'Monthly average certificate-of-deposit rate.' in text
    table_row = next(line for line in text.splitlines() if line.startswith('| Certificates'))
    assert '2026-07 versus 2026-04' in table_row
    assert '90-day' not in table_row


def test_country_source_clocks_and_verified_metadata_names_are_retained(inputs):
    series = inputs[0]['series'][0]
    series.update(source='SSB_NATIVE', available_at='2026-09-10T22:30:00+00:00',
                  publisher_updated_at='2026-09-10T06:00:00+00:00',
                  publisher_metadata={'producer': 'Statistics Norway'})
    snapshot = build(inputs)
    text = render_country(snapshot, 'NO')
    assert 'Statistics Norway' in text and '2026-09-10T22:30:00+00:00' in text
    assert 'Dataset updated: 2026-09-10T06:00:00+00:00' in text
    assert 'Publication timestamp: not supplied' in text
    index = render_index(snapshot)
    assert '2026-09-10T22:30:00+00:00' in index
    assert 'Sweden retains its earlier eligible capture' in index


def test_source_text_and_links_are_safe_and_unknown_countries_rejected(inputs):
    inputs[0]['series'][0]['definition'] = '<script>bad</script> | [unsafe](javascript:bad)'
    snapshot = build(inputs)
    text = render_country(snapshot, 'NO')
    assert '<script>' not in text and '&lt;script&gt;' in text and r'\[unsafe\]' in text
    with pytest.raises(ValueError, match='country'):
        render_country(snapshot, '../bad')
    ref = snapshot['countries'][1]['signals'][0]['evidence_refs'][0]
    snapshot['citations'][ref]['source_url'] = 'javascript:bad'
    with pytest.raises(ValueError, match='URL'):
        render_country(reseal(snapshot), 'NO')


def test_render_preserves_all_input_bytes_and_is_deterministic(inputs):
    snapshot = build(inputs)
    original = deepcopy(snapshot)
    assert render_index(snapshot) == render_index(snapshot)
    assert render_country(snapshot, 'SE') == render_country(snapshot, 'SE')
    assert snapshot == original
    snapshot['countries'][0]['signals'][0]['latest']['value'] = 0
    with pytest.raises(ValueError, match='hash'):
        render_country(snapshot, 'SE')
