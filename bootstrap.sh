#!/bin/bash
# Headless pilot run: deps, data, train, eval. Vast --onstart-cmd calls this.
# Logs tee'd to disk; DONE only on full success (watcher destroys box on DONE).
# ponytail: no set -e — prep_data.py crashes at interpreter teardown (numpy
# in-place upgrade) AFTER writing files. Explicit checks instead.
cd "$(dirname "$0")"

pip install -U pip || exit 1
pip install "numpy>=2.0" "pyarrow>=14" || exit 1
pip install "transformers>=5.0.0" accelerate trl datasets unsloth \
    huggingface_hub hf_transfer lm-eval || exit 1

python -u prep_data.py > prep_detail.log 2>&1 || true
tail -n 5 prep_detail.log | tee prep.log
test -f train_vi.jsonl || { echo PREP_FAILED; exit 1; }
python -u train_pilot.py 2>&1 | tee train.log; test ${PIPESTATUS[0]} -eq 0 || { echo TRAIN_FAILED; exit 1; }
python eval_gate.py lfm350m-vi-cpt-pilot 2>&1 | tee eval.log; test ${PIPESTATUS[0]} -eq 0 || { echo EVAL_FAILED; exit 1; }
touch DONE
echo ALLDONE
