#!/bin/bash
cd /workspace/lfm350m-vi
BEST_VI=999
declare -A LRS=( [1]=2e-4 [2]=1e-4 [3]=5e-5 )
declare -A DIRS=( [1]=lfm350m-vi-cpt-full [2]=lfm350m-vi-cpt-r2 [3]=lfm350m-vi-cpt-r3 )
declare -A LOGS=(  [1]=train2.log      [2]=train_r2.log      [3]=train_r3.log )

# round 1 already running on box — just wait for it
while pgrep -f "train_pilot.py --max-steps 4000" > /dev/null; do sleep 120; done
sleep 15

for ROUND in 1 2 3; do
    DIR=${DIRS[$ROUND]}
    if [ ! -f "$DIR/adapter_model.safetensors" ]; then
        echo "ROUND${ROUND}_NOADAPTER" >> RESULT
        continue
    fi
    python3 -u eval_full.py "$DIR" > "eval_round${ROUND}.log" 2>&1
    VI=$(grep -oE 'tuned: vi=[0-9.]+' "eval_round${ROUND}.log" | grep -oE '[0-9.]+$' | head -1)
    EN=$(grep -oE 'en=[0-9.]+' "eval_round${ROUND}.log" | tail -1 | grep -oE '[0-9.]+')
    echo "R${ROUND} vi=${VI:-NA} en=${EN:-NA}" >> RESULT
    if [ -n "$VI" ] && python3 -c "import sys; sys.exit(0 if float('$VI') < $BEST_VI else 1)"; then
        BEST_VI=$VI
        echo "$DIR" > BEST_DIR
    fi
    if [ -n "$VI" ] && python3 -c "import sys; vi=float('$VI'); en=float('${EN:-999}'); sys.exit(0 if vi < 5.0 and en < 40 else 1)"; then
        echo "GOOD at round $ROUND vi=$VI en=$EN best=$DIR" >> RESULT
        touch DONE2
        exit 0
    fi
    if [ "$ROUND" -lt 3 ]; then
        NEXT=$((ROUND + 1))
        echo "BAD round $ROUND vi=$VI — launching round $NEXT from $DIR lr=${LRS[$NEXT]}" >> RESULT
        nohup python3 -u train_pilot.py --max-steps 4000 \
            --output-dir "${DIRS[$NEXT]}" --lr "${LRS[$NEXT]}" \
            --init-adapter "$DIR" > "${LOGS[$NEXT]}" 2>&1 &
        while pgrep -f "train_pilot.py --max-steps 4000" > /dev/null; do sleep 120; done
        sleep 15
    fi
done
echo "BAD exhausted rounds best_vi=$BEST_VI best_dir=$(cat BEST_DIR)" >> RESULT
touch DONE2
