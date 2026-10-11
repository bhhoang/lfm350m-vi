#!/bin/bash
# Headless pilot run: deps, data, train, eval. Vast --onstart-cmd calls this.
# Logs tee'd to disk; DONE signals completion for auto-destroy watcher.
set -e
cd "$(dirname "$0")"

pip install -U pip
pip install "numpy>=2.0" "pyarrow>=14"
pip install "transformers>=5.0.0" accelerate trl datasets unsloth \
    huggingface_hub hf_transfer lm-eval

python -u prep_data.py > prep_detail.log 2>&1; PREP_RC=$?
tail -n 5 prep_detail.log | tee prep.log
# ponytail: numpy in-place upgrade crashes interpreter teardown after output
# done. Data files written = success regardless of teardown RC.
test -f train_vi.jsonl || exit $PREP_RC
python -u train_pilot.py 2>&1 | tee train.log; test ${PIPESTATUS[0]} -eq 0
python eval_gate.py lfm350m-vi-cpt-pilot 2>&1 | tee eval.log
touch DONE
echo ALLDONE
