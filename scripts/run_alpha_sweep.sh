#!/bin/bash
# Alpha sensitivity sweep for Modifications A and B
# α ∈ {0.01, 0.1, 0.5, 1.0} x 3 seeds on cartpole_swingup (500K frames)
#
# Usage:
#   bash scripts/run_alpha_sweep.sh
#   bash scripts/run_alpha_sweep.sh --dry-run

set -e

TASK="cartpole_swingup"
ALPHAS=(0.01 0.1 0.5 1.0)
SEEDS=(1 2 3)
FRAMES=500000
PYTHON="${PYTHON:-d:/CSC415_Project/drqv2/venv/Scripts/python.exe}"
DRY_RUN=false

for arg in "$@"; do
    case $arg in
        --dry-run) DRY_RUN=true ;;
    esac
done

TOTAL_RUNS=$(( 2 * ${#ALPHAS[@]} * ${#SEEDS[@]} ))
RUN_IDX=0

echo "============================================"
echo "  Alpha Sensitivity Sweep"
echo "============================================"
echo "Task:    $TASK ($FRAMES frames)"
echo "Alphas:  ${ALPHAS[*]}"
echo "Seeds:   ${SEEDS[*]}"
echo "Total:   $TOTAL_RUNS runs"
echo "============================================"

# ---- Sweep Modification A ----
echo ""
echo "=== Modification A (Consistency Regularizer) ==="
for alpha in "${ALPHAS[@]}"; do
    for s in "${SEEDS[@]}"; do
        RUN_IDX=$((RUN_IDX + 1))
        EXP="alpha_sweep_modA_a${alpha}_s${s}"
        CMD="\"$PYTHON\" train.py task@_global_=$TASK \
            seed=$s num_train_frames=$FRAMES \
            use_consistency=true consistency_alpha=$alpha \
            experiment=$EXP save_video=false replay_buffer_num_workers=1"

        echo "[$RUN_IDX/$TOTAL_RUNS] $EXP"
        if $DRY_RUN; then
            echo "  CMD: $CMD"
        else
            eval $CMD
            echo "  Done."
        fi
    done
done

# ---- Sweep Modification B ----
echo ""
echo "=== Modification B (Task-Conditional Contrastive) ==="
for alpha in "${ALPHAS[@]}"; do
    for s in "${SEEDS[@]}"; do
        RUN_IDX=$((RUN_IDX + 1))
        EXP="alpha_sweep_modB_a${alpha}_s${s}"
        CMD="\"$PYTHON\" train.py task@_global_=$TASK \
            seed=$s num_train_frames=$FRAMES \
            use_contrastive=true contrastive_alpha=$alpha \
            experiment=$EXP save_video=false replay_buffer_num_workers=1"

        echo "[$RUN_IDX/$TOTAL_RUNS] $EXP"
        if $DRY_RUN; then
            echo "  CMD: $CMD"
        else
            eval $CMD
            echo "  Done."
        fi
    done
done

echo ""
echo "============================================"
if $DRY_RUN; then
    echo "Dry run complete. $TOTAL_RUNS commands printed."
else
    echo "All $TOTAL_RUNS alpha sweep runs complete."
fi
echo "============================================"
