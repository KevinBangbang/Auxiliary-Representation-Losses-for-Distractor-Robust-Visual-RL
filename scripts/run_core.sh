#!/bin/bash
# Core experiment subset: cartpole_swingup only, seed=1 first
# ~16 runs, ~40 hours total on RTX 5090 Laptop
#
# Usage: nohup bash scripts/run_core.sh > exp_local/run_core.log 2>&1 &

set -e
cd "$(dirname "$0")/.."

PYTHON="venv/Scripts/python.exe"
LOG="exp_local/run_core_progress.txt"
mkdir -p exp_local
echo "$$" > exp_local/run_core.pid

TASK="cartpole_swingup"
FRAMES=500000
SEEDS=(1)  # Start with seed=1; add 2 3 later to fill 3 seeds

run_exp() {
    local EXP="$1"; shift
    echo "" | tee -a "$LOG"
    echo "[$(date +%H:%M:%S)] START $EXP" | tee -a "$LOG"
    if "$PYTHON" train.py "$@" "hydra.run.dir=./exp_local/\${now:%Y.%m.%d}/${EXP}" 2>&1 | tail -3; then
        echo "[$(date +%H:%M:%S)] OK    $EXP" | tee -a "$LOG"
    else
        echo "[$(date +%H:%M:%S)] FAIL  $EXP" | tee -a "$LOG"
    fi
}

echo "============================================" | tee -a "$LOG"
echo "  Core Subset: $TASK, seeds=${SEEDS[*]}" | tee -a "$LOG"
echo "  Started: $(date)" | tee -a "$LOG"
echo "============================================" | tee -a "$LOG"

# ---- Phase 1: Clean (4 methods x seeds) ----
echo "" | tee -a "$LOG"
echo "=== Clean Environment ===" | tee -a "$LOG"
for method in "baseline:false:false" "modA:true:false" "modB:false:true" "modAB:true:true"; do
    IFS=':' read -r name cons cont <<< "$method"
    for s in "${SEEDS[@]}"; do
        EXP="${TASK}_clean_${name}_s${s}"
        ARGS=("task@_global_=$TASK" "seed=$s" "num_train_frames=$FRAMES"
              "experiment=$EXP" "save_video=false" "replay_buffer_num_workers=4" "save_snapshot=true")
        [ "$cons" = "true" ] && ARGS+=("use_consistency=true")
        [ "$cont" = "true" ] && ARGS+=("use_contrastive=true")
        run_exp "$EXP" "${ARGS[@]}"
    done
done

# ---- Phase 2: Distractor (4 methods x seeds) ----
echo "" | tee -a "$LOG"
echo "=== Distractor Environment ===" | tee -a "$LOG"
for method in "baseline:false:false" "modA:true:false" "modB:false:true" "modAB:true:true"; do
    IFS=':' read -r name cons cont <<< "$method"
    for s in "${SEEDS[@]}"; do
        EXP="${TASK}_distractor_${name}_s${s}"
        ARGS=("task@_global_=$TASK" "seed=$s" "num_train_frames=$FRAMES"
              "experiment=$EXP" "save_video=false" "replay_buffer_num_workers=4"
              "save_snapshot=true" "use_distractors=true")
        [ "$cons" = "true" ] && ARGS+=("use_consistency=true")
        [ "$cont" = "true" ] && ARGS+=("use_contrastive=true")
        run_exp "$EXP" "${ARGS[@]}"
    done
done

# ---- Phase 3: Alpha sweep (2 mods x 4 alphas x seeds) ----
echo "" | tee -a "$LOG"
echo "=== Alpha Sensitivity Sweep ===" | tee -a "$LOG"

for alpha in 0.01 0.1 0.5 1.0; do
    for s in "${SEEDS[@]}"; do
        EXP="alpha_modA_a${alpha}_s${s}"
        run_exp "$EXP" "task@_global_=$TASK" "seed=$s" "num_train_frames=$FRAMES" \
            "experiment=$EXP" "save_video=false" "replay_buffer_num_workers=4" \
            "save_snapshot=true" "use_consistency=true" "consistency_alpha=$alpha"
    done
done

for alpha in 0.01 0.1 0.5 1.0; do
    for s in "${SEEDS[@]}"; do
        EXP="alpha_modB_a${alpha}_s${s}"
        run_exp "$EXP" "task@_global_=$TASK" "seed=$s" "num_train_frames=$FRAMES" \
            "experiment=$EXP" "save_video=false" "replay_buffer_num_workers=4" \
            "save_snapshot=true" "use_contrastive=true" "contrastive_alpha=$alpha"
    done
done

# ---- Phase 4: Plots ----
echo "" | tee -a "$LOG"
echo "=== Generating Plots ===" | tee -a "$LOG"
"$PYTHON" scripts/plot_experiment_results.py

echo "" | tee -a "$LOG"
echo "============================================" | tee -a "$LOG"
echo "  CORE SUBSET COMPLETE" | tee -a "$LOG"
echo "  Finished: $(date)" | tee -a "$LOG"
echo "============================================" | tee -a "$LOG"
