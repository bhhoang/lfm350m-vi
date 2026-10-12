#!/bin/bash
# Headless pilot run: deps, data, train, eval. Vast --onstart-cmd calls this.
# Logs tee'd to disk; DONE only on full success (watcher destroys box on DONE).
# ponytail: no set -e — prep_data.py crashes at interpreter teardown (numpy
# in-place upgrade) AFTER writing files. Explicit checks instead.
# ponytail: python3 not python — vastai image has no python symlink.
cd "$(dirname "$0")"
PY=python3

$PY -m pip install "numpy>=2.0" "pyarrow>=14" || exit 1
# ponytail: unsloth resolves a working torch/tv pair.
$PY -m pip install unsloth || exit 1
# ponytail: stale torchaudio breaks trl import (ABI vs new torch).
$PY -m pip uninstall -y torchaudio || true
$PY -m pip install "fsspec>=2023.1.0,<=2026.2.0" || exit 1
$PY -m pip install "transformers>=4.52.4,<=5.17.0" "trl>=0.18.2,<=1.13.0" \
    accelerate datasets peft huggingface_hub hf_transfer lm-eval || exit 1

$PY -u prep_data.py > prep_detail.log 2>&1 || true
tail -n 5 prep_detail.log | tee prep.log
test -f train_vi.jsonl || { echo PREP_FAILED; exit 1; }
$PY -u train_pilot.py 2>&1 | tee train.log; test ${PIPESTATUS[0]} -eq 0 || { echo TRAIN_FAILED; exit 1; }
$PY eval_gate.py lfm350m-vi-cpt-pilot 2>&1 | tee eval.log; test ${PIPESTATUS[0]} -eq 0 || { echo EVAL_FAILED; exit 1; }
touch DONE
echo ALLDONE
