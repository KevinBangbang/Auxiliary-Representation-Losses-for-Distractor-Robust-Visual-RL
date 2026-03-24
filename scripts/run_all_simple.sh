#!/bin/bash
# Simple sequential experiment runner
# Usage: bash scripts/run_all_simple.sh 2>&1 | tee exp_local/run_all.log

set -e
cd "$(dirname "$0")/.."

PYTHON="venv/Scripts/python.exe"
LOG="exp_local/run_all_progress.txt"
mkdir -p exp_local

echo "============================================" | tee -a "$LOG"
echo "  DrQ-v2 Full Experiment Suite" | tee -a "$LOG"
echo "  Started: $(date)" | tee -a "$LOG"
echo "============================================" | tee -a "$LOG"

run_exp() {
    local EXP="$1"; shift
    echo "" | tee -a "$LOG"
    echo "[$(date +%H:%M:%S)] START $EXP" | tee -a "$LOG"
    # Use short hydra.run.dir to avoid Windows path-length issues with multiprocess workers
    if "$PYTHON" train.py "$@" "hydra.run.dir=./exp_local/\${now:%Y.%m.%d}/${EXP}" 2>&1 | tail -3; then
        echo "[$(date +%H:%M:%S)] OK    $EXP" | tee -a "$LOG"
    else
        echo "[$(date +%H:%M:%S)] FAIL  $EXP" | tee -a "$LOG"
    fi
}

# =============================================
# Phase 1: Clean experiments (36 runs)
# =============================================
echo ""
echo "=== Phase 1: Clean Environment ===" | tee -a "$LOG"

# cartpole_swingup 500K
for method in "baseline:false:false" "modA:true:false" "modB:false:true" "modAB:true:true"; do
    IFS=':' read -r name cons cont <<< "$method"
    for s in 1 2 3; do
        EXP="cartpole_swingup_clean_${name}_s${s}"
        ARGS=("task@_global_=cartpole_swingup" "seed=$s" "num_train_frames=500000"
              "experiment=$EXP" "save_video=false" "replay_buffer_num_workers=4" "save_snapshot=true")
        [ "$cons" = "true" ] && ARGS+=("use_consistency=true")
        [ "$cont" = "true" ] && ARGS+=("use_contrastive=true")
        run_exp "$EXP" "${ARGS[@]}"
    done
done

# walker_walk 1M
for method in "baseline:false:false" "modA:true:false" "modB:false:true" "modAB:true:true"; do
    IFS=':' read -r name cons cont <<< "$method"
    for s in 1 2 3; do
        EXP="walker_walk_clean_${name}_s${s}"
        ARGS=("task@_global_=walker_walk" "seed=$s" "num_train_frames=1000000"
              "experiment=$EXP" "save_video=false" "replay_buffer_num_workers=4" "save_snapshot=true")
        [ "$cons" = "true" ] && ARGS+=("use_consistency=true")
        [ "$cont" = "true" ] && ARGS+=("use_contrastive=true")
        run_exp "$EXP" "${ARGS[@]}"
    done
done

# cheetah_run 1M
for method in "baseline:false:false" "modA:true:false" "modB:false:true" "modAB:true:true"; do
    IFS=':' read -r name cons cont <<< "$method"
    for s in 1 2 3; do
        EXP="cheetah_run_clean_${name}_s${s}"
        ARGS=("task@_global_=cheetah_run" "seed=$s" "num_train_frames=1000000"
              "experiment=$EXP" "save_video=false" "replay_buffer_num_workers=4" "save_snapshot=true")
        [ "$cons" = "true" ] && ARGS+=("use_consistency=true")
        [ "$cont" = "true" ] && ARGS+=("use_contrastive=true")
        run_exp "$EXP" "${ARGS[@]}"
    done
done

# =============================================
# Phase 2: Distractor experiments (36 runs)
# =============================================
echo ""
echo "=== Phase 2: Distractor Environment ===" | tee -a "$LOG"

for task_spec in "cartpole_swingup:500000" "walker_walk:1000000" "cheetah_run:1000000"; do
    IFS=':' read -r task frames <<< "$task_spec"
    for method in "baseline:false:false" "modA:true:false" "modB:false:true" "modAB:true:true"; do
        IFS=':' read -r name cons cont <<< "$method"
        for s in 1 2 3; do
            EXP="${task}_distractor_${name}_s${s}"
            ARGS=("task@_global_=$task" "seed=$s" "num_train_frames=$frames"
                  "experiment=$EXP" "save_video=false" "replay_buffer_num_workers=4"
                  "save_snapshot=true" "use_distractors=true")
            [ "$cons" = "true" ] && ARGS+=("use_consistency=true")
            [ "$cont" = "true" ] && ARGS+=("use_contrastive=true")
            run_exp "$EXP" "${ARGS[@]}"
        done
    done
done

# =============================================
# Phase 3: Alpha sensitivity sweep (24 runs)
# =============================================
echo ""
echo "=== Phase 3: Alpha Sweep ===" | tee -a "$LOG"

# Mod A alpha sweep
for alpha in 0.01 0.1 0.5 1.0; do
    for s in 1 2 3; do
        EXP="alpha_modA_a${alpha}_s${s}"
        run_exp "$EXP" "task@_global_=cartpole_swingup" "seed=$s" "num_train_frames=500000" \
            "experiment=$EXP" "save_video=false" "replay_buffer_num_workers=4" \
            "use_consistency=true" "consistency_alpha=$alpha"
    done
done

# Mod B alpha sweep
for alpha in 0.01 0.1 0.5 1.0; do
    for s in 1 2 3; do
        EXP="alpha_modB_a${alpha}_s${s}"
        run_exp "$EXP" "task@_global_=cartpole_swingup" "seed=$s" "num_train_frames=500000" \
            "experiment=$EXP" "save_video=false" "replay_buffer_num_workers=4" \
            "use_contrastive=true" "contrastive_alpha=$alpha"
    done
done

# =============================================
# Phase 4: Generate plots
# =============================================
echo ""
echo "=== Phase 4: Generating Plots ===" | tee -a "$LOG"
"$PYTHON" scripts/plot_experiment_results.py

echo ""
echo "============================================" | tee -a "$LOG"
echo "  ALL EXPERIMENTS COMPLETE" | tee -a "$LOG"
echo "  Finished: $(date)" | tee -a "$LOG"
echo "============================================" | tee -a "$LOG"
