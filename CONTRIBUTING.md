# Working on Frazil

## Branches

`main` always works: every script in the docs runs on it. Everything else happens on a short-lived branch, named for what it does:

| prefix | for | example |
| --- | --- | --- |
| `feat/` | new capability | `feat/thickness-members` |
| `fix/` | a bug | `fix/iss-001-report-run-tags` |
| `exp/` | a modelling experiment (a training run and its evaluation) | `exp/radiation-channels` |
| `chore/` | housekeeping, docs, tooling | `chore/housekeeping` |
| `docs/` | documentation only | `docs/credits` |

Start from an up-to-date `main`:

```bash
git switch main && git pull && git switch -c feat/short-name
```

Commit on the branch as you go. When it works, merge it back with a pull request on GitHub (it records the diff and the reasoning, even when working alone), or locally:

```bash
git switch main && git merge --no-ff feat/short-name && git branch -d feat/short-name
```

An experiment that didn't pan out is still worth merging if its tooling or its write-up is useful; otherwise delete the branch and note the result in [docs/ISSUES.md](docs/ISSUES.md).

## Tests

```bash
pytest            # fast tier: no data needed, ~40 s
pytest --slow     # also end-to-end runs on real data and models, ~5 min (run before merging to main)
```

The fast tier runs every script's self-check, loads every script from an unrelated folder, compares each script's defaults and every config recipe with stored snapshots (`tests/snapshots/`), forbids scripts importing each other or hard-coding paths, and unit-tests the `frazil` package. The slow tier runs a free-run with its map and members, rebuilds the forcing bit-for-bit, checks the zero-initialised channel expansion against its seed model, and renders each animation frame type.

- **Run the fast tier automatically before every push** (once per clone): `git config core.hooksPath .githooks`. Skip once with `git push --no-verify`.
- **Every bug fix comes with a test that fails without the fix.** That is how the earlier regressions would have been caught.
- **Intended changes to defaults or configs:** `pytest --update-snapshots`, then commit the snapshot diff with the change, so the review shows exactly what moved.

## Commits

One logical change per commit, message as `type: what changed` (`add:`, `update:`, `fix:`, `remove:`, `docs:`), with a short body saying why when it isn't obvious. Refer to issue IDs ("closes ISS-001").

## Keeping the record

- **Changes:** add a line under *Unreleased* in [CHANGELOG.md](CHANGELOG.md) in the same branch as the change.
- **Issues:** new problems and ideas go in [docs/ISSUES.md](docs/ISSUES.md) with the next ID; close them there when fixed.
- **Experiments:** each gets a folder in [experiments/](experiments/README.md) (copy `_template.md`) and a line in its index; training recipes go in `configs/experiment/`.
- **Results:** runs log themselves to `results/experiments.jsonl` with the code fingerprint; pass `--note` so the report says what each run was.
- **Releases:** see *Versions and releases* below; the plan is in [docs/ROADMAP.md](docs/ROADMAP.md).

## Versions and releases

Versions follow [semantic versioning](https://semver.org/): `vMAJOR.MINOR.PATCH`. The version lives in three places that must agree: `setup.py`, the heading in [CHANGELOG.md](CHANGELOG.md), and an annotated git tag `vX.Y.Z`. (The `version_1` tag in a local clone is GenSIM's own, fetched from the original repository.)

**What changes which number.** Before 1.0, anything may still change:
- **MINOR** (`0.3.0`): a roadmap milestone, meaning new features, a new recommended model, or a breaking change. Breaking changes are listed under *Changed* with what to do about them.
- **PATCH** (`0.3.1`): bug fixes and documentation. Outputs don't change except where the fix requires it.

From 1.0 on, a breaking change needs a **MAJOR** version. Breaking means any of:
- a script, flag or default renamed, removed or changed in meaning
- output names changed
- the `frazil` package's functions changed incompatibly
- a training recipe changed so it trains a different model
- checkpoints or datacubes no longer loading
- the scorecard protocol changed

When renaming a flag, keep the old one working (with a warning) for one minor version.

**Models in releases.** Each release names its recommended model in [docs/DATA_AND_MODELS.md](docs/DATA_AND_MODELS.md), with its scorecard and the SHA-256 of its checkpoint (`shasum -a 256 <ckpt>`), because checkpoints aren't in git.

**Release checklist**
1. Everything planned for the milestone is merged; `pytest` and `pytest --slow` pass on `main`.
2. If the recommended model changed: its scorecard is recorded, its targets are met or explicitly waived in the release notes, and the guardrails hold. The README's *What it can and can't do* numbers come from that scorecard.
3. Move *Unreleased* in the changelog under `## [X.Y.Z] — date`; set the same version in `setup.py`; tick the milestone in [docs/ROADMAP.md](docs/ROADMAP.md).
4. Commit as `release: vX.Y.Z`, then tag and push:
   ```bash
   git tag -a vX.Y.Z -m "Frazil X.Y.Z: <one-line summary>" && git push origin main vX.Y.Z
   ```
5. Optionally publish a GitHub release with the changelog section as its notes.

## Fixing bugs

**Severity**

| | what it means | when |
| --- | --- | --- |
| **S1 critical** | wrong results without any error, lost data, the main workflow (preprocess, train, free-run, animate) crashing, or an already-published result affected | fix now; patch release as soon as it's merged |
| **S2 major** | a tool crashes or gives wrong output, but there's a workaround | next patch release |
| **S3 minor** | cosmetic, documentation, rough edges | whenever convenient |

**Route**
1. Record it in [docs/ISSUES.md](docs/ISSUES.md): kind *bug*, its severity, how to reproduce it.
2. Branch from `main`: `fix/iss-NNN-short-name`.
3. **Write the test first:** a test that fails because of the bug. It stays in the suite so the bug can't come back.
4. Fix it. Run `pytest`, and `pytest --slow` if the fix touches models, data or outputs.
5. Add a *Fixed* line to the changelog's *Unreleased* section, citing the issue; move the issue to *Closed*.
6. Merge. For S1 and S2, follow the release checklist for a patch version.

**Errata.** If a bug affected results that were already written up (an `experiments/` entry, the README numbers, a scorecard, a rendered artwork), add a dated *Erratum* note to each affected place saying what was wrong and what changed. Re-run the affected analysis if its conclusion could change, and mention it in the changelog.

## Data and models

Every default data, model and output location lives in `frazil/paths.py`; a new dataset or output folder gets a constant there, not a string in a script. Shared code goes in the `frazil/` package; scripts never import each other.


Data (`data/`) and outputs (`plots/`) are never committed. A trained model is identified by its run folder in `data/models/<name>/`, which holds the exact resolved config it was trained with (`.hydra/config.yaml`); the results ledger links evaluations to the code that produced them.
