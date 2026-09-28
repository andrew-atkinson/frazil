#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Append-only experiment ledger (JSONL) with code fingerprinting.

Every run appends ONE JSON record to results/experiments.jsonl:
  timestamp, command, params, git state, and structured metrics (per variable,
  across leads). Notes and a project-start entry go in the same ledger.

Tying results to code BEFORE you commit: each record stores the base commit plus
a `worktree_sha256` fingerprint of the exact uncommitted changes (tracked diff +
untracked files by content hash), and archives the tracked diff to
results/patches/<sha>.patch. So a result made on a dirty tree is still pinned to
a reproducible code state. When you later commit, run
`results_log.py note "committed <sha>: ..."` to bookmark the link (the fingerprint
and patch already let you reconstruct exactly what ran).

Deterministic: records the RNG seed, writes canonical JSON (sorted keys, rounded
floats), and derives run_id from the record content.

CLI:
  python experiments/results_log.py init "first tracked results; pushforward added"
  python experiments/results_log.py note "20k pushforward fine-tune complete"
  python experiments/results_log.py show
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEDGER = os.path.join(ROOT, "results", "experiments.jsonl")
PATCHES = os.path.join(ROOT, "results", "patches")



def model_tag(ckpt):
    """Short, filesystem-safe model id from a checkpoint path, for naming outputs:
    monthly.ckpt -> 'monthly'; monthly_pf/last.ckpt -> 'monthly_pf';
    monthly_pf/step_10000.ckpt -> 'monthly_pf_step_10000'."""
    import re
    stem = os.path.splitext(os.path.basename(ckpt))[0]
    parent = os.path.basename(os.path.dirname(ckpt))
    if re.match(r"^(last|best)(-v\d+)?$", stem):      # generic Lightning names
        m = parent
    elif stem.startswith("step_"):
        m = f"{parent}_{stem}"
    else:
        m = stem
    return re.sub(r"[^A-Za-z0-9._-]+", "_", m)

def _git(*args):
    try:
        return subprocess.run(["git", *args], capture_output=True, cwd=ROOT).stdout
    except Exception:
        return b""


def _gits(*args):
    return _git(*args).decode(errors="replace").strip()


def git_state(archive=True):
    """Base commit + a fingerprint of the exact working tree (so a dirty-tree
    result is still pinned to reproducible code)."""
    commit = _gits("rev-parse", "HEAD") or None
    branch = _gits("rev-parse", "--abbrev-ref", "HEAD") or None
    diff = _git("diff", "HEAD")                              # tracked modifications
    # Exclude the ledger's own results/ tree so the fingerprint is a pure CODE
    # hash: identical code -> identical worktree_sha256 (deterministic).
    untracked = [f for f in _gits("ls-files", "--others", "--exclude-standard").splitlines()
                 if f and not f.startswith("results/")]
    unt = [{"file": f, "sha": _gits("hash-object", f)} for f in untracked]
    fp = hashlib.sha256(
        diff + b"".join((u["file"] + " " + u["sha"] + "\n").encode() for u in unt)
    ).hexdigest()
    patch_file = None
    if archive and diff:
        os.makedirs(PATCHES, exist_ok=True)
        patch_file = os.path.join("results", "patches", f"{fp[:12]}.patch")
        with open(os.path.join(ROOT, patch_file), "wb") as fh:
            fh.write(diff)
    return {"branch": branch, "commit": commit, "dirty": bool(diff) or bool(unt),
            "worktree_sha256": fp, "untracked": unt, "patch_file": patch_file}


def _canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _round(x, nd=4):
    if isinstance(x, float):
        return round(x, nd)
    if isinstance(x, dict):
        return {k: _round(v, nd) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_round(v, nd) for v in x]
    return x


def append(record: dict) -> str:
    record.setdefault(
        "timestamp",
        datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    record.setdefault("git", git_state())
    record["run_id"] = hashlib.sha256(
        _canonical({k: v for k, v in record.items() if k != "run_id"}).encode()
    ).hexdigest()[:12]
    os.makedirs(os.path.dirname(LEDGER), exist_ok=True)
    with open(LEDGER, "a") as fh:
        fh.write(_canonical(record) + "\n")
    return record["run_id"]


def log_result(experiment, params, metrics, note=None, command=None) -> str:
    """Append a structured result. `metrics` is a dict of per-variable arrays
    across leads (see rollout_monthly for the shape)."""
    rec = {
        "type": "result",
        "experiment": experiment,
        "command": command or ("python " + " ".join(sys.argv)),
        "params": _round(params),
        "metrics": _round(metrics),
    }
    if note:
        rec["note"] = note
    rid = append(rec)
    dirty = rec.get("git", {}).get("dirty")
    print(f"[log] {experiment} -> {os.path.relpath(LEDGER, ROOT)}  run_id={rid}"
          + ("  (dirty tree fingerprinted)" if dirty else ""))
    return rid


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("init", "note"):
        sub.add_parser(name).add_argument("text")
    sub.add_parser("show")
    args = ap.parse_args(argv)

    if args.cmd in ("init", "note"):
        rid = append({"type": "project" if args.cmd == "init" else "note",
                      "text": args.text})
        print(f"[log] {args.cmd} recorded  run_id={rid} -> {os.path.relpath(LEDGER, ROOT)}")
    elif args.cmd == "show":
        if not os.path.exists(LEDGER):
            print("(no ledger yet)")
            return
        for line in open(LEDGER):
            r = json.loads(line)
            g = r.get("git", {})
            tag = (g.get("commit") or "nogit")[:7] + ("+dirty" if g.get("dirty") else "")
            head = r.get("experiment") or r.get("type")
            msg = r.get("text") or r.get("note") or ""
            print(f"{r['timestamp']}  {tag:14s}  {head:18s}  {r.get('run_id')}"
                  + (f"  {msg}" if msg else ""))


if __name__ == "__main__":
    main()
