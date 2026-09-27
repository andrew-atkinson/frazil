#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Figure for the README: each dataset's time span and its influence on the model.

Colour = the dataset's role (training target / forcing / observations).
Fill = influence: solid shapes the trained weights; hatched is used only at run
time or for scoring. Background bands mark the training, validation,
out-of-sample test and projection periods. Writes a light and a dark version
(docs/figures/data_timeline_{light,dark}.png); the README picks by theme.

Spans are the data actually in the repo (see docs/DATA_AND_MODELS.md).

Run:  python experiments/plot_data_timeline.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Validated categorical slots 1-3 (dataviz reference palette), light and dark steps.
THEME = {
    "light": dict(surface="#fcfcfb", text="#0b0b0b", muted="#52514e", band=("#f0efec", "#e6e5e1"),
                  role={"target": "#2a78d6", "forcing": "#eb6834", "obs": "#1baf7a"}),
    "dark": dict(surface="#1a1a19", text="#ffffff", muted="#c3c2b7", band=("#252524", "#30302e"),
                 role={"target": "#3987e5", "forcing": "#d95926", "obs": "#199e70"}),
}
ROLE_NAME = {"target": "training target (what the model learns to produce)",
             "forcing": "forcing (what drives the model)",
             "obs": "observations (correction and scoring)"}

# (label, role, [(start_year, end_year_exclusive, solid?)], influence note)
ROWS = [
    ("neXtSIM-OPA sea-ice states", "target", [(1995, 2019, True)],
     "Total: the only source of the model's ice physics"),
    ("QM-corrected targets (sic)", "target", [(1995, 2019, True)],
     "Partial: sic only, in the QM models"),
    ("ERA5 air temperature, humidity, winds", "forcing", [(1994, 2019, True)],
     "Total: every model's atmosphere (1994 = spin-up)"),
    ("ERA5 extension", "forcing", [(2018, 2026, False)],
     "Run time only: the 2019–25 out-of-sample hindcast"),
    ("ERA5 SST, solar and thermal radiation", "forcing", [(1995, 2019, True), (2019, 2026, False)],
     "Partial: only the SST / radiation models"),
    ("CMIP6 MPI-ESM1-2-LR ssp245", "forcing", [(2015, 2101, False)],
     "Run time only: projections; never trained on"),
    ("NSIDC-0051 concentration (gridded)", "obs", [(1995, 2019, True), (2019, 2026, False)],
     "Indirect: fits the QM targets (to 2018); scores 2019–25"),
    ("NSIDC Sea Ice Index (extent)", "obs", [(1979, 2026, False)],
     "Evaluation only"),
]
PERIODS = [(1995, 2015, "training"), (2015, 2019, "val."), (2019, 2026, "test"),
           (2026, 2101, "projection (CMIP only)")]


def draw(mode, path):
    th = THEME[mode]
    plt.rcParams.update({"font.size": 10, "text.color": th["text"], "axes.labelcolor": th["muted"],
                         "xtick.color": th["muted"], "ytick.color": th["text"]})
    fig, ax = plt.subplots(figsize=(14, 5.2), facecolor=th["surface"])
    ax.set_facecolor(th["surface"])
    n = len(ROWS)
    for i, (x0, x1, name) in enumerate(PERIODS):
        ax.axvspan(x0, x1, color=th["band"][i % 2], zorder=0, lw=0)
        ax.text((x0 + x1) / 2, n - 0.35, name, ha="center", va="bottom", color=th["muted"], fontsize=9)
    for r, (label, role, segs, note) in enumerate(ROWS):
        y = n - 1 - r
        c = th["role"][role]
        for x0, x1, solid in segs:
            ax.barh(y, x1 - x0, left=x0, height=0.56, zorder=2,
                    color=c if solid else th["surface"], edgecolor=c,
                    hatch=None if solid else "////", linewidth=1.4)
        ax.text(1.01, y, note, transform=ax.get_yaxis_transform(), va="center",
                color=th["muted"], fontsize=9)
    ax.set_yticks(range(n))
    ax.set_yticklabels([r[0] for r in ROWS][::-1])
    ax.set_xlim(1976, 2102)
    ax.set_ylim(-0.7, n + 0.3)
    ax.set_xticks([1980, 1995, 2015, 2026, 2050, 2075, 2100])
    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Datasets: time span and influence on the model", loc="left",
                 fontsize=13, color=th["text"], pad=14)
    role_h = [Patch(facecolor=th["role"][k], edgecolor=th["role"][k], label=v) for k, v in ROLE_NAME.items()]
    fill_h = [Patch(facecolor=th["muted"], edgecolor=th["muted"], label="solid: shapes the trained weights"),
              Patch(facecolor=th["surface"], edgecolor=th["muted"], hatch="////",
                    label="hatched: run time or scoring only")]
    # matplotlib fills legend columns top-down: order so row 1 = roles, row 2 = fills
    handles = [role_h[0], fill_h[0], role_h[1], fill_h[1], role_h[2]]
    leg = fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
                     fontsize=9, bbox_to_anchor=(0.42, -0.01))
    for t in leg.get_texts():
        t.set_color(th["text"])
    fig.subplots_adjust(left=0.235, right=0.71, top=0.86, bottom=0.2)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=160, facecolor=th["surface"])
    plt.close(fig)
    print(f"[plot] {path}")


if __name__ == "__main__":
    for mode in THEME:
        draw(mode, f"docs/figures/data_timeline_{mode}.png")
