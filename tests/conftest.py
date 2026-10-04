"""Shared test setup.

Two tiers:
  pytest            fast checks; no data or trained models needed (run before every push)
  pytest --slow     also the end-to-end checks that need data/ and data/models/ (run before merging)

Snapshots of script defaults and composed configs live in tests/snapshots/. When a
change to them is intended, regenerate with `pytest --update-snapshots` and commit
the diff alongside the change.
"""
import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SNAPSHOTS = os.path.join(ROOT, "tests", "snapshots")


def pytest_addoption(parser):
    parser.addoption("--slow", action="store_true", help="also run tests that need data and trained models")
    parser.addoption("--update-snapshots", action="store_true", help="rewrite stored snapshots instead of comparing")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--slow"):
        return
    skip = pytest.mark.skip(reason="needs data and models: run with --slow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def update_snapshots(request):
    return request.config.getoption("--update-snapshots")


def check_snapshot(name, value, update):
    """Compare `value` (JSON-able) with tests/snapshots/<name>.json, or rewrite it."""
    path = os.path.join(SNAPSHOTS, f"{name}.json")
    if update or not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(value, fh, indent=1, sort_keys=True, default=str)
            fh.write("\n")
        if not update:
            pytest.fail(f"no snapshot yet; wrote {os.path.relpath(path, ROOT)} - review and commit it")
        return
    with open(path) as fh:
        stored = json.load(fh)
    assert json.loads(json.dumps(value, default=str)) == stored, (
        f"{name} differs from its snapshot; if intended, run pytest --update-snapshots")


# Runs a script only until argparse has parsed its arguments, then prints them.
_DUMP = r"""
import argparse, json, runpy, sys
script, extra = sys.argv[1], sys.argv[2:]
orig = argparse.ArgumentParser.parse_args
def grab(self, args=None, namespace=None):
    ns = orig(self, extra, namespace)
    print("DEFAULTS:" + json.dumps(vars(ns), default=str, sort_keys=True)); raise SystemExit(0)
argparse.ArgumentParser.parse_args = grab
sys.argv = [script] + extra
runpy.run_path(script, run_name="__main__")
"""


def script_defaults(script, cwd, extra=()):
    """Parse a script's arguments from `cwd` (not the repo root) and return its defaults."""
    env = dict(os.environ, PYTORCH_ENABLE_MPS_FALLBACK="1")
    p = subprocess.run([sys.executable, "-c", _DUMP, os.path.join(ROOT, script), *extra],
                       cwd=cwd, capture_output=True, text=True, env=env, timeout=300)
    lines = [l for l in p.stdout.splitlines() if l.startswith("DEFAULTS:")]
    assert p.returncode == 0 and lines, f"{script} failed from {cwd}:\n{p.stderr[-2000:]}"
    return json.loads(lines[-1][len("DEFAULTS:"):])


def run(args, cwd=ROOT, timeout=1800):
    """Run a command with the test interpreter; return the CompletedProcess (asserting success)."""
    env = dict(os.environ, PYTORCH_ENABLE_MPS_FALLBACK="1")
    p = subprocess.run([sys.executable, *args], cwd=cwd, capture_output=True, text=True, env=env, timeout=timeout)
    assert p.returncode == 0, f"{' '.join(args)} failed:\n{p.stdout[-1500:]}\n{p.stderr[-2500:]}"
    return p
