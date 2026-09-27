#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Turn the results ledger (results/experiments.jsonl) into readable comparison
tables -- so you can see every run and how checkpoints stack up at a glance,
instead of scrolling terminal history.

Writes results/REPORT.md (Markdown tables) and prints a compact summary. Re-run
after any experiment to refresh it.

Run:  python experiments/results_report.py
"""
from __future__ import annotations

import json
import os
import statistics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEDGER = os.path.join(ROOT, "results", "experiments.jsonl")
REPORT = os.path.join(ROOT, "results", "REPORT.md")
STATES = ["sit", "sic", "sid", "siu", "siv", "snt"]


def model_of(params):
    c = params.get("ckpt_resolved", "") or ""
    if "monthly_pf" in c:
        return "monthly_pf"
    if c.endswith("monthly.ckpt"):
        return "monthly"
    return os.path.splitext(os.path.basename(c))[0] or "?"


def _mean_skill_at(rec, key, lead):
    m = rec["metrics"]
    sk = m.get(key)
    if not sk or lead not in m.get("leads_months", []):
        return None
    i = m["leads_months"].index(lead)
    return statistics.mean(sk[v][i] for v in STATES) * 100


def load():
    if not os.path.exists(LEDGER):
        raise SystemExit(f"no ledger at {LEDGER} -- run some experiments first")
    return [json.loads(l) for l in open(LEDGER)]


def main():
    recs = load()
    out = [f"# Experiment results report",
           f"\n_generated from {os.path.relpath(LEDGER, ROOT)} "
           f"({len(recs)} records)_\n"]

    # 1. Every run, newest first.
    out.append("## Runs\n")
    out.append("| when (UTC) | experiment | model | note | run_id |")
    out.append("|---|---|---|---|---|")
    for r in sorted(recs, key=lambda r: r["timestamp"], reverse=True):
        note = r.get("note") or r.get("text") or ""
        exp = r.get("experiment") or r.get("type")
        model = model_of(r.get("params", {})) if r.get("params") else ""
        out.append(f"| {r['timestamp'][:16]} | {exp} | {model} | {note[:48]} | `{r['run_id']}` |")

    # 2. rollout: skill vs climatology, mean over the 6 vars, at key leads.
    roll = [r for r in recs if r.get("experiment") == "rollout_monthly"]
    if roll:
        out.append("\n## rollout_monthly — skill vs climatology (mean of 6 vars, %)\n")
        out.append("| model | note | 1mo | 3mo | 6mo | 12mo |")
        out.append("|---|---|---|---|---|---|")
        for r in roll:
            cells = []
            for L in (1, 3, 6, 12):
                s = _mean_skill_at(r, "skill_vs_climatology", L)
                cells.append(f"{s:+.0f}" if s is not None else "–")
            out.append(f"| {model_of(r['params'])} | {r.get('note','')[:24]} | "
                       + " | ".join(cells) + " |")

    # 3. eval: one-step skill vs persistence, mean over vars.
    ev = [r for r in recs if r.get("experiment") == "eval_monthly"]
    if ev:
        out.append("\n## eval_monthly — one-step skill vs persistence (%)\n")
        out.append("| model | note | mean skill | beats |")
        out.append("|---|---|---|---|")
        for r in ev:
            sp = r["metrics"].get("skill_vs_persistence", {})
            ms = statistics.mean(sp.values()) * 100 if sp else None
            out.append(f"| {model_of(r['params'])} | {r.get('note','')[:24]} | "
                       f"{ms:+.0f} | {r['metrics'].get('beats_persistence','?')}/6 |")

    # 4. freerun: drift stability.
    fr = [r for r in recs if r.get("experiment") == "freerun_monthly"]
    if fr:
        out.append("\n## freerun_monthly — drift (higher retention / smaller drift = better)\n")
        out.append("| model | note | yrs | sharp SIT retention | area drift (Mkm²) | non-finite |")
        out.append("|---|---|---|---|---|---|")
        for r in fr:
            m = r["metrics"]
            ad = (m["ice_area_km2"]["final"] - m["ice_area_km2"]["initial"]) / 1e6
            out.append(f"| {model_of(r['params'])} | {r.get('note') or '':.24} | "
                       f"{m.get('years_completed','?')} | "
                       f"{m.get('sharpness_sit_retention', float('nan')):.3f} | "
                       f"{ad:+.2f} | {m.get('nonfinite_total','?')} |")

    md = "\n".join(out) + "\n"
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w") as fh:
        fh.write(md)
    print(md)
    print(f"[report] -> {os.path.relpath(REPORT, ROOT)}")


if __name__ == "__main__":
    main()
