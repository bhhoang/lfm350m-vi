#!/bin/bash
# Headless pilot run: deps, data, train, eval. Vast --onstart-cmd calls this.
# Logs tee'd to disk; DONE signals completion for auto-destroy watcher.
set -e
cd "$(dirname "$0")"

pip install -U pip
pip install "transformers>=5.0.0" accelerate trl datasets unsloth \
    huggingface_hub hf_transfer lm-eval

python -u prep_data.py 2>&1 | tee prep.log
python -u train_pilot.py 2>&1 | tee train.log
python eval_gate.py lfm350m-vi-cpt-pilot 2>&1 | tee eval.log
touch DONE
echo ALLDONE
