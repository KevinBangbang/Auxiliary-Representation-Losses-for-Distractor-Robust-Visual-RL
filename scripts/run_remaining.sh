#!/bin/bash
# Run only experiments that haven't completed yet
# Checks eval.csv line count to determine if done

set -e
cd "$(dirname "$0")/.."

PYTHON="venv/Scripts/python.exe"
LOG="exp_local/run_remaining_progress.txt"

is_done() {
    local EXP="$1"
    local NEED_LINES="$2"  # expected eval.csv lines (101 for 1M, 51 for 500K)
    # Search all date dirs for this experiment
    for d in exp_local/2026.*/; do
        local ev="$d$EXP/eval.csv"
        if [ -f "$ev" ]; then
            local lines=$(wc -l < "$ev")
            if [ "$lines" -ge "$NEED_LINES" ]; then
                return 0  # done
            fi
        fi
        # Also check long hydra dir names
        for hd in "$d"*"experiment=${EXP},"*/eval.csv; do
            if [ -f "$hd" ]; then
                local lines2=$(wc -l < "$hd")
                if [ "$lines2" -ge "$NEED_LINES" ]; then
                    return 0
                fi
            fi
        done
    done
    return 1  # not done
}

run_if_needed() {
    local EXP="$1"
    local NEED_LINES="$2"
    shift 2
    if is_done "$EXP" "$NEED_LINES"; then
        echo "[$(date +%H:%M:%S)] SKIP  $EXP (already done)" | tee -a "$LOG"
        return
    fi
    echo "[$(date +%H:%M:%S)] START $EXP" | tee -a "$LOG"
    if "$PYTHON" train.py "$@" "hydra.run.dir=./exp_local/\${now:%Y.%m.%d}/${EXP}" 2>&1 | tail -3; then
        echo "[$(date +%H:%M:%S)] OK    $EXP" | tee -a "$LOG"
    else
        echo "[$(date +%H:%M:%S)] FAIL  $EXP" | tee -a "$LOG"
    fi
}

echo "============================================" | tee -a "$LOG"
echo "  Remaining Experiments" | tee -a "$LOG"
echo "  Started: $(date)" | tee -a "$LOG"
echo "============================================" | tee -a "$LOG"

# --- Clean walker_walk (1M, need 101 lines) ---
for method in "baseline:false:false" "modA:true:false" "modB:false:true" "modAB:true:true"; do
    IFS=':' read -r name cons cont <<< "$method"
    for s in 1; do
        EXP="walker_walk_clean_${name}_s${s}"
        ARGS=("task@_global_=walker_walk" "seed=$s" "num_train_frames=1000000"
              "experiment=$EXP" "save_video=false" "replay_buffer_num_workers=0"
              "save_snapshot=true" "replay_buffer_size=500000")
        [ "$cons" = "true" ] && ARGS+=("use_consistency=true")
        [ "$cont" = "true" ] && ARGS+=("use_contrastive=true")
        run_if_needed "$EXP" 101 "${ARGS[@]}"
    done
done

# --- Clean cheetah_run (1M, need 101 lines) ---
for method in "baseline:false:false" "modA:true:false" "modB:false:true" "modAB:true:true"; do
    IFS=':' read -r name cons cont <<< "$method"
    for s in 1; do
        EXP="cheetah_run_clean_${name}_s${s}"
        ARGS=("task@_global_=cheetah_run" "seed=$s" "num_train_frames=1000000"
              "experiment=$EXP" "save_video=false" "replay_buffer_num_workers=0"
              "save_snapshot=true" "replay_buffer_size=500000")
        [ "$cons" = "true" ] && ARGS+=("use_consistency=true")
        [ "$cont" = "true" ] && ARGS+=("use_contrastive=true")
        run_if_needed "$EXP" 101 "${ARGS[@]}"
    done
done

# --- Distractor walker_walk (1M, need 101 lines) ---
for method in "baseline:false:false" "modA:true:false" "modB:false:true" "modAB:true:true"; do
    IFS=':' read -r name cons cont <<< "$method"
    for s in 1; do
        EXP="walker_walk_distractor_${name}_s${s}"
        ARGS=("task@_global_=walker_walk" "seed=$s" "num_train_frames=1000000"
              "experiment=$EXP" "save_video=false" "replay_buffer_num_workers=0"
              "save_snapshot=true" "use_distractors=true" "replay_buffer_size=500000")
        [ "$cons" = "true" ] && ARGS+=("use_consistency=true")
        [ "$cont" = "true" ] && ARGS+=("use_contrastive=true")
        run_if_needed "$EXP" 101 "${ARGS[@]}"
    done
done

# --- Distractor cheetah_run (1M, need 101 lines) ---
for method in "baseline:false:false" "modA:true:false" "modB:false:true" "modAB:true:true"; do
    IFS=':' read -r name cons cont <<< "$method"
    for s in 1; do
        EXP="cheetah_run_distractor_${name}_s${s}"
        ARGS=("task@_global_=cheetah_run" "seed=$s" "num_train_frames=1000000"
              "experiment=$EXP" "save_video=false" "replay_buffer_num_workers=0"
              "save_snapshot=true" "use_distractors=true" "replay_buffer_size=500000")
        [ "$cons" = "true" ] && ARGS+=("use_consistency=true")
        [ "$cont" = "true" ] && ARGS+=("use_contrastive=true")
        run_if_needed "$EXP" 101 "${ARGS[@]}"
    done
done

echo "" | tee -a "$LOG"
echo "============================================" | tee -a "$LOG"
echo "  ALL REMAINING EXPERIMENTS COMPLETE" | tee -a "$LOG"
echo "  Finished: $(date)" | tee -a "$LOG"
echo "============================================" | tee -a "$LOG"
