"""DOT builder for pressure chains (slice P3)."""
from dalio.app.fundamentals.dot import chain_to_dot
from dalio.app.fundamentals.snapshot import Chain, parse_snapshot


def _chain(**over) -> Chain:
    base = dict(
        iso2="US", rule_id="fiscal_dominance", title="Fiscal dominance", triggered=True, severity=0.96,
        constraint='Debt 124 % of GDP with deficit -6.8 % · r > g',
        forced_options=("monetize / inflate", "financial repression", "slow austerity"),
        spillovers=(("CN", "real return on USD reserves falls", "via rates"),
                    ("SA", "USD peg imports inflation", "via FX"),
                    ("SA", "second text same target", "via FX"),
                    ("domestic banks", "sovereign-bank loop", "via debt")),
        confidence=0.7,
        inputs={"gov_debt_pct_gdp": 124.0, "fiscal_balance_pct_gdp": -6.8,
                "interest_burden_pct_gdp": 3.6, "gdp_growth_fwd5": None},
        uncertainty="C",
    )
    base.update(over)
    return Chain(**base)


def test_dot_structure_and_escaping(synthetic_snapshot_dict):
    snap = parse_snapshot(synthetic_snapshot_dict)
    names = dict(zip(snap.players["iso2"], snap.players["name"], strict=True))
    dot = chain_to_dot(_chain(constraint='Debt "124" % · r > g'), names, snap.catalog)
    assert dot.startswith("digraph chain {") and dot.rstrip().endswith("}")
    assert dot.count("in") >= 4                                   # 4 inputs incl. the None one
    assert 'in3 [label="Real growth, next 5 y\\n—"]' in dot       # None input drawn with em dash
    assert "GDP per capita" not in dot
    assert 'penwidth=2' in dot and '\\"124\\"' in dot             # constraint node + escaped quotes
    assert dot.count(" -> c;") == 4
    assert dot.count("c -> o") == 3
    # 3 distinct targets (SA de-duplicated), dashed, labelled with the channel
    assert dot.count('style="dashed", color=') == 3
    assert '[label="via rates", style="dashed"]' in dot and '[label="via FX", style="dashed"]' in dot
    assert '"China"' in dot and '"domestic banks"' in dot          # player name resolved, group label kept
    assert "severity 96%" in dot and "confidence 70%" in dot and "tier C" in dot


def test_dot_isolation_inputs():
    dot = chain_to_dot(_chain(rule_id="isolation", title="Isolation", forced_options=("capital controls",),
                              spillovers=(("US", "x", "via rates"),),
                              inputs={"political_stability_pct": 10.0, "sanctioned": 1.0}), {"US": "United States"})
    assert "in0 [label=\"political stability\\n10.0\"]" in dot
    assert 'in1 [label="Sanctions\\nin force"]' in dot
    assert dot.count(" -> c;") == 2


def test_dot_without_options_links_constraint_to_targets():
    dot = chain_to_dot(_chain(forced_options=(), spillovers=(("CN", "x", ""),)), {"CN": "China"})
    assert "c -> s0" in dot and "c -> o" not in dot
