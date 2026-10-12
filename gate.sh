#!/bin/bash
cd /workspace/lfm350m-vi
while pgrep -f "train_pilot.py --max-steps 4000" > /dev/null; do sleep 120; done
sleep 15
if [ ! -f lfm350m-vi-cpt-full/adapter_model.safetensors ]; then
    echo TRAIN2_FAILED > RESULT; touch DONE2; exit 1
fi
python3 -u eval_full.py lfm350m-vi-cpt-full > eval_full2.log 2>&1
VI=$(grep -oE 'tuned: vi=[0-9.]+' eval_full2.log | grep -oE '[0-9.]+$')
EN=$(grep -oE 'en=[0-9.]+' eval_full2.log | tail -1 | grep -oE '[0-9.]+')
if [ -z "$VI" ]; then
    echo EVAL_FAILED > RESULT; touch DONE2; exit 1
fi
if python3 -c "import sys; vi=float('$VI'); en=float('${EN:-999}'); sys.exit(0 if vi < 5.8 and en < 45 else 1)"; then
    echo GOOD > RESULT
else
    echo BAD > RESULT
fi
touch DONE2
