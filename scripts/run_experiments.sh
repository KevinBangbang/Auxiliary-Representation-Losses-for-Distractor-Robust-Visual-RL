#!/bin/bash
# Full experiment matrix for DrQ-v2 + Auxiliary Losses
# 6 settings (3 tasks x {clean, distractor}) x 4 methods x 3 seeds = 72 runs
#
# Usage:
#   bash scripts/run_experiments.sh                  # run all
#   bash scripts/run_experiments.sh --dry-run        # print commands only
#   bash scripts/run_experiments.sh --clean-only     # clean environments only
#   bash scripts/run_experiments.sh --distractor-only # distractor environments only

set -e

# ---- Configuration ----
TASKS=("cartpole_swingup" "walker_walk" "cheetah_run")
SEEDS=(1 2 3)
PYTHON="${PYTHON:-d:/CSC415_Project/drqv2/venv/Scripts/python.exe}"
VIDEO_DIR="${VIDEO_DIR:-d:/CSC415_Project/drqv2/kinetics_videos}"
DRY_RUN=false
RUN_CLEAN=true
RUN_DISTRACTOR=true

# Task-specific frame budgets
declare -A TASK_FRAMES
TASK_FRAMES[cartpole_swingup]=500000
TASK_FRAMES[walker_walk]=1000000
TASK_FRAMES[cheetah_run]=1000000

# Method definitions: name:use_consistency:use_contrastive
METHODS=(
    "baseline:false:false"
    "modA:true:false"
    "modB:false:true"
    "modAB:true:true"
)

# ---- Parse arguments ----
for arg in "$@"; do
    case $arg in
        --dry-run) DRY_RUN=true ;;
        --clean-only) RUN_DISTRACTOR=false ;;
        --distractor-only) RUN_CLEAN=false ;;
    esac
done

# ---- Build environment list ----
ENVS=()
if $RUN_CLEAN; then ENVS+=("clean"); fi
if $RUN_DISTRACTOR; then ENVS+=("distractor"); fi

# ---- Count total runs ----
TOTAL_RUNS=$(( ${#ENVS[@]} * ${#TASKS[@]} * ${#METHODS[@]} * ${#SEEDS[@]} ))

echo "============================================"
echo "  DrQ-v2 Auxiliary Loss Experiments"
echo "============================================"
echo "Tasks:        ${TASKS[*]}"
echo "Methods:      baseline, modA, modB, modAB"
echo "Environments: ${ENVS[*]}"
echo "Seeds:        ${SEEDS[*]}"
echo "Total runs:   $TOTAL_RUNS"
echo ""
for task in "${TASKS[@]}"; do
    echo "  $task: ${TASK_FRAMES[$task]} frames"
done
echo "============================================"
if $DRY_RUN; then
    echo "[DRY RUN] Only printing commands."
    echo ""
fi

# ---- Run experiments ----
RUN_IDX=0
for env_type in "${ENVS[@]}"; do
    for task in "${TASKS[@]}"; do
        frames=${TASK_FRAMES[$task]}
        for method_spec in "${METHODS[@]}"; do
            IFS=':' read -r method_name use_cons use_cont <<< "$method_spec"
            for s in "${SEEDS[@]}"; do
                RUN_IDX=$((RUN_IDX + 1))
                EXP_NAME="${task}_${env_type}_${method_name}_s${s}"

                DISTRACTOR_FLAGS=""
                if [ "$env_type" = "distractor" ]; then
                    DISTRACTOR_FLAGS="use_distractors=true distractor_video_dir=$VIDEO_DIR"
                fi

                CMD="\"$PYTHON\" train.py task@_global_=$task \
                    seed=$s num_train_frames=$frames \
                    use_consistency=$use_cons use_contrastive=$use_cont \
                    $DISTRACTOR_FLAGS \
                    experiment=$EXP_NAME \
                    save_video=false save_train_video=false \
                    replay_buffer_num_workers=1"

                echo "[$RUN_IDX/$TOTAL_RUNS] $EXP_NAME"

                if $DRY_RUN; then
                    echo "  CMD: $CMD"
                    echo ""
                else
                    echo "  Starting..."
                    eval $CMD
                    echo "  Done."
                    echo ""
                fi
            done
        done
    done
done

echo "============================================"
if $DRY_RUN; then
    echo "Dry run complete. $TOTAL_RUNS commands printed."
else
    echo "All $TOTAL_RUNS runs complete."
fi
echo "============================================"
