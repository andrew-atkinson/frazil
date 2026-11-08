#!/bin/zsh
# All the no-training runs, one after another. Resumable: re-run the same command.
# Run from the repo root:  caffeinate -i zsh experiments/2026-11-05_loose_ice/run_tests.sh
set -e
PY=/opt/homebrew/Caskroom/miniforge/base/envs/frazil/bin/python
S=experiments/2026-11-05_loose_ice/step_test.py
M=data/models; D=data/train_data; O=$M/loose_ice_tests
V04=$M/monthly_pf_qm_ema_qmw/last.ckpt
$PY $S --ckpt $V04 --datacube $D/monthly_datacube_sicqmw9 --out $O/v04_truth
$PY $S --ckpt $V04 --datacube $D/monthly_datacube_sicqmw9 --out $O/v04_generated --start generated
$PY $S --ckpt $V04 --datacube $D/monthly_datacube_sicqmw9 --out $O/v04_truth_n40 --n-steps 40
$PY $S --ckpt $M/monthly_qm/last.ckpt --datacube $D/monthly_datacube_sicqm --out $O/qm_nopf_truth
$PY $S --ckpt $M/monthly_pf_qm/last.ckpt --datacube $D/monthly_datacube_sicqm --out $O/qm_pf_truth
$PY $S --ckpt $V04 --datacube $D/monthly_datacube_sicqmw9 --out $O/v04_truth_noclamp --no-clamp
