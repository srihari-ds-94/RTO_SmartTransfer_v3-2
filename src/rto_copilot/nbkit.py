"""Small helpers shared by the notebooks: Pride palette, light chart style, loaders and CI bar charts.

Keeping this here (not in every notebook) keeps each notebook cell short enough to explain on screen.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import config
from .scoring import wilson_interval

NAVY, TEAL, MINT = "#0F2A4A", "#0E8C6F", "#D6EEE6"
GREY, AMBER, RED, SKY = "#8A97A5", "#D98E04", "#C0392B", "#6FA8DC"
PALETTE = [NAVY, TEAL, SKY, AMBER, GREY, RED]


def style():
    """Light, minimal matplotlib style (Calibri/Cambria fall back to DejaVu if not installed)."""
    import logging
    import matplotlib as mpl
    from matplotlib import font_manager
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    installed = {f.name for f in font_manager.fontManager.ttflist}
    fonts = [f for f in ("Calibri", "Carlito", "DejaVu Sans") if f in installed] or ["sans-serif"]
    mpl.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": "#cfd8e0",
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True, "grid.color": "#eef2f5",
        "axes.titleweight": "bold", "axes.titlecolor": NAVY, "axes.labelcolor": "#44546a",
        "xtick.color": "#44546a", "ytick.color": "#44546a", "font.size": 10, "figure.dpi": 110,
        "axes.prop_cycle": mpl.cycler(color=PALETTE), "legend.frameon": False,
        "font.family": fonts,
    })


def load_jsonl(path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def agent_run(name: str) -> list[dict]:
    """Scored rows of a saved agent run, e.g. agent_run('w3_dev_final')."""
    return load_jsonl(config.RESULTS / "agent" / name / "answers.jsonl")


def ci_bars(ax, labels, successes, totals, colors=None, horizontal=False, fmt="{:.0%}"):
    """Bar chart of proportions with Wilson 95% whiskers and the value printed on top."""
    rates = [s / t if t else 0 for s, t in zip(successes, totals)]
    cis = [wilson_interval(s, t) for s, t in zip(successes, totals)]
    lo = [r - c[0] for r, c in zip(rates, cis)]
    hi = [c[1] - r for r, c in zip(rates, cis)]
    colors = colors or [TEAL] * len(labels)
    if horizontal:
        ax.barh(labels, rates, xerr=[lo, hi], color=colors, error_kw={"ecolor": GREY, "capsize": 3})
        ax.set_xlim(0, 1.05)
        for i, (r, t) in enumerate(zip(rates, totals)):
            ax.text(min(r + hi[i] + 0.02, 1.0), i, f"{fmt.format(r)} (n={t})", va="center", fontsize=9)
    else:
        ax.bar(labels, rates, yerr=[lo, hi], color=colors, error_kw={"ecolor": GREY, "capsize": 3})
        ax.set_ylim(0, 1.05)
        for i, (r, t) in enumerate(zip(rates, totals)):
            ax.text(i, min(r + hi[i] + 0.02, 1.0), f"{fmt.format(r)}\nn={t}", ha="center", fontsize=9)
    return ax


def two_prop_min_detectable(n1: int, n2: int, p: float = 0.5, alpha_z: float = 1.96, power_z: float = 0.84) -> float:
    """Smallest true difference between two pass rates that a two-sided 5% test detects with 80% power (normal approx)."""
    return (alpha_z + power_z) * (p * (1 - p) * (1 / n1 + 1 / n2)) ** 0.5
