"""Pressure chain → Graphviz DOT (pure string building; rendered client-side by
``st.graphviz_chart`` — verified in P0: no ``dot`` binary, no ``graphviz`` pip).

Rank columns, left to right:

    inputs (value · threshold)  →  binding constraint  →  forced options  →  spillover targets
                                   (rust, penwidth 2)                        (dashed; edge label = channel)
"""
from __future__ import annotations

from collections.abc import Mapping

from dalio.app.fundamentals.snapshot import Chain, IndicatorMeta
from dalio.app.theme import INK, INK_MUTED, RUST

FONT = "Inter Tight"


def _esc(s: str) -> str:
    """Escape for a DOT double-quoted label. Real newlines become DOT ``\\n``;
    a literal backslash-n already in the text is left as the line break it is."""
    return str(s).replace('"', '\\"').replace("\n", "\\n")


def _fmt_input(name: str, value: float | None, catalog: Mapping[str, IndicatorMeta]) -> str:
    meta = catalog.get(name)
    label = meta.label if meta else name.replace("_", " ")
    if value is None:
        return f"{label}\\n—"
    unit = f" {meta.unit}" if meta and meta.unit and len(meta.unit) < 14 else ""
    v = f"{value:,.0f}" if abs(value) >= 1000 else f"{value:.1f}"
    return f"{label}\\n{v}{unit}"


def chain_to_dot(
    chain: Chain,
    player_names: Mapping[str, str],
    catalog: Mapping[str, IndicatorMeta] | None = None,
) -> str:
    """One graph per fired rule. Inputs that were ``None`` are still drawn (with
    an em dash) so the reader sees what the rule looked at."""
    catalog = catalog or {}
    title = _esc(f"{chain.title} · severity {chain.severity:.0%} · confidence {chain.confidence:.0%} · tier C")
    lines = [
        "digraph chain {",
        '  rankdir=LR; bgcolor="transparent"; nodesep=0.25; ranksep=0.6;',
        f'  graph [label="{title}", labelloc=t, labeljust=l, fontname="{FONT}", fontsize=11, fontcolor="{INK_MUTED}"];',
        f'  node [shape=box, style="filled", fillcolor="transparent", fontname="{FONT}", fontsize=10, '
        f'color="{INK}", fontcolor="{INK}", margin="0.12,0.06"];',
        f'  edge [color="{INK}", fontname="{FONT}", fontsize=9, fontcolor="{INK_MUTED}", arrowsize=0.7];',
    ]
    # inputs (skip the static/flag entries the rules stash in inputs)
    inputs = [(k, v) for k, v in chain.inputs.items()
              if k not in ("fx_regime", "sanctioned", "dsr_q90", "political_stability_pct")]
    if chain.rule_id == "isolation":
        pv = chain.inputs.get("political_stability_pct")
        inputs = [("political_stability", None if pv is None else float(pv))]
        if chain.inputs.get("sanctioned"):
            inputs.append(("sanctioned", None))
    for i, (name, value) in enumerate(inputs):
        label = _fmt_input(name, value, catalog) if name != "sanctioned" else "Sanctions\\nin force"
        lines.append(f'  in{i} [label="{_esc(label)}"];')
    lines.append(f'  c [label="{_esc(chain.constraint)}", penwidth=2, color="{RUST}", fontcolor="{INK}"];')
    for i, opt in enumerate(chain.forced_options):
        lines.append(f'  o{i} [label="{_esc(opt)}"];')
    # spillovers: one node per distinct target, edge label = channel(s)
    seen: dict[str, list[str]] = {}
    for target, text, channel in chain.spillovers:
        seen.setdefault(target, []).append(channel or text)
    for j, target in enumerate(seen):
        name = player_names.get(target, target)
        lines.append(f'  s{j} [label="{_esc(name)}", style="dashed", color="{INK_MUTED}", fontcolor="{INK}"];')
    for i in range(len(inputs)):
        lines.append(f"  in{i} -> c;")
    for i in range(len(chain.forced_options)):
        lines.append(f"  c -> o{i};")
    n_opts = max(1, len(chain.forced_options))
    for j, (_target, channels) in enumerate(seen.items()):
        src = f"o{j % n_opts}" if chain.forced_options else "c"
        label = _esc(channels[0]) if channels else ""
        lines.append(f'  {src} -> s{j} [label="{label}", style="dashed"];')
    lines.append("}")
    return "\n".join(lines)
