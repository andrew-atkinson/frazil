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

## Commits

One logical change per commit, message as `type: what changed` (`add:`, `update:`, `fix:`, `remove:`, `docs:`), with a short body saying why when it isn't obvious. Refer to issue IDs ("closes ISS-001").

## Keeping the record

- **Changes:** add a line under *Unreleased* in [CHANGELOG.md](CHANGELOG.md) in the same branch as the change.
- **Issues:** new problems and ideas go in [docs/ISSUES.md](docs/ISSUES.md) with the next ID; close them there when fixed.
- **Experiments:** each gets a folder in [experiments/](experiments/README.md) (copy `_template.md`) and a line in its index; training recipes go in `configs/experiment/`.
- **Results:** runs log themselves to `results/experiments.jsonl` with the code fingerprint; pass `--note` so the report says what each run was.
- **Releases:** when `main` reaches a milestone, move *Unreleased* under a version and tag it: `git tag -a v0.2.0 -m "..." && git push --tags`.

## Data and models

Data (`data/`) and outputs (`plots/`) are never committed. A trained model is identified by its run folder in `data/models/<name>/`, which holds the exact resolved config it was trained with (`.hydra/config.yaml`); the results ledger links evaluations to the code that produced them.
