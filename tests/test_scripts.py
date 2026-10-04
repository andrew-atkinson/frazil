"""Fast checks over every script in scripts/: they load and parse from anywhere,
their defaults don't drift, their self-checks pass, and they stay independent."""
import ast
import glob
import importlib.util
import os

import pytest

from conftest import ROOT, check_snapshot, run, script_defaults

SCRIPTS = sorted(os.path.relpath(p, ROOT) for p in glob.glob(os.path.join(ROOT, "scripts", "*", "*.py")))
SELFCHECK = [s for s in SCRIPTS if '"--selfcheck"' in open(os.path.join(ROOT, s)).read()]
POSITIONAL = {"scripts/evaluate/channel_weights.py": ["x.ckpt"]}   # required arguments


@pytest.mark.parametrize("script", SELFCHECK)
def test_selfcheck(script):
    run([script, "--selfcheck"])


@pytest.mark.parametrize("script", SCRIPTS)
def test_loads_from_any_folder_and_defaults_unchanged(script, tmp_path, update_snapshots):
    """Run from an unrelated folder (catches imports that only resolve from the repo
    or the script's own folder) and compare the parsed defaults with the snapshot."""
    defaults = script_defaults(script, cwd=tmp_path, extra=POSITIONAL.get(script, []))
    check_snapshot(f"defaults/{os.path.splitext(os.path.basename(script))[0]}", defaults, update_snapshots)


def _imports(path):
    for n in ast.walk(ast.parse(open(path).read())):
        if isinstance(n, ast.Import):
            yield n.lineno, [a.name for a in n.names]
        elif isinstance(n, ast.ImportFrom) and n.module:
            yield n.lineno, [n.module]


def test_no_script_imports_another_script():
    names = {os.path.splitext(os.path.basename(s))[0] for s in SCRIPTS}
    bad = [f"{s}:{line} imports {m}" for s in SCRIPTS for line, mods in _imports(os.path.join(ROOT, s))
           for m in mods if m.split(".")[0] in names]
    assert not bad, "shared code belongs in frazil/:\n" + "\n".join(bad)


def test_no_shared_path_hardcoded_outside_paths_module():
    spec = importlib.util.spec_from_file_location("paths", os.path.join(ROOT, "frazil", "paths.py"))
    paths = importlib.util.module_from_spec(spec); spec.loader.exec_module(paths)
    shared = {v for k, v in vars(paths).items() if k.isupper()}
    files = SCRIPTS + [os.path.relpath(p, ROOT) for p in glob.glob(os.path.join(ROOT, "frazil", "*.py"))] + ["train.py"]
    bad = []
    for f in files:
        if f.endswith("paths.py"):
            continue
        tree = ast.parse(open(os.path.join(ROOT, f)).read())
        docs = {id(n.body[0].value) for n in ast.walk(tree)
                if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef)) and n.body
                and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)}
        bad += [f"{f}:{n.lineno} {n.value!r}" for n in ast.walk(tree)
                if isinstance(n, ast.Constant) and n.value in shared and id(n) not in docs]
    assert not bad, "use frazil.paths instead of:\n" + "\n".join(bad)


def test_no_script_locates_files_from_its_own_folder():
    """Scripts move between folders; anything they need to find is defined in frazil
    (paths.py, ledger.LEDGER), which knows where the repository is."""
    bad = [f"{s}:{n.lineno}" for s in SCRIPTS for n in ast.walk(ast.parse(open(os.path.join(ROOT, s)).read()))
           if isinstance(n, ast.Name) and n.id == "__file__"]
    assert not bad, "use frazil.paths / frazil.ledger instead of __file__:\n" + "\n".join(bad)
