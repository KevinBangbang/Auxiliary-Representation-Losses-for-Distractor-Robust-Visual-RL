"""
One-script runner for ALL project experiments.

Runs in order:
  1. Main comparison: clean (3 tasks x 4 methods x 3 seeds = 36 runs)
  2. Main comparison: distractor (3 tasks x 4 methods x 3 seeds = 36 runs)
  3. Alpha sensitivity sweep (2 mods x 4 alphas x 3 seeds = 24 runs)
  4. Plot results

Features:
  - Skips experiments whose eval.csv already exists
  - Saves progress log to exp_local/run_all_log.txt
  - Ctrl+C gracefully stops after current experiment

Usage:
  python scripts/run_all.py                     # run everything
  python scripts/run_all.py --clean-only        # clean experiments only
  python scripts/run_all.py --distractor-only   # distractor only
  python scripts/run_all.py --alpha-only        # alpha sweep only
  python scripts/run_all.py --dry-run           # print commands, don't run
  python scripts/run_all.py --skip-distractor   # skip distractor experiments
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# ---- Configuration ----

TASKS = ["cartpole_swingup", "walker_walk", "cheetah_run"]
TASK_FRAMES = {
    "cartpole_swingup": 500000,
    "walker_walk": 1000000,
    "cheetah_run": 1000000,
}
SEEDS = [1, 2, 3]
METHODS = [
    ("baseline", False, False),
    ("modA", True, False),
    ("modB", False, True),
    ("modAB", True, True),
]
ALPHAS = [0.01, 0.1, 0.5, 1.0]

PROJECT_DIR = Path(__file__).parent.parent.resolve()
PYTHON = str(PROJECT_DIR / "venv" / "Scripts" / "python.exe")
VIDEO_DIR = str(PROJECT_DIR / "kinetics_videos")

# Graceful stop
_stop_requested = False
def _signal_handler(sig, frame):
    global _stop_requested
    print("\n[!] Stop requested. Will finish current experiment then exit.")
    _stop_requested = True
signal.signal(signal.SIGINT, _signal_handler)


def find_existing_eval(task, method, env_type, seed):
    """Check if an experiment with matching config already has eval.csv."""
    exp_root = PROJECT_DIR / "exp_local"
    if not exp_root.exists():
        return False
    for date_dir in exp_root.iterdir():
        if not date_dir.is_dir():
            continue
        for exp_dir in date_dir.iterdir():
            if not exp_dir.is_dir():
                continue
            config_path = exp_dir / ".hydra" / "config.yaml"
            eval_path = exp_dir / "eval.csv"
            if not config_path.exists() or not eval_path.exists():
                continue
            # Quick check: look for experiment name in the dir name
            exp_name = f"{task}_{env_type}_{method}_s{seed}"
            if exp_name in exp_dir.name:
                # Verify eval.csv has reasonable data
                try:
                    with open(eval_path) as f:
                        lines = f.readlines()
                    if len(lines) > 5:  # at least a few eval points
                        return True
                except:
                    pass
    return False


def run_experiment(task, method_name, use_cons, use_cont, seed, env_type,
                   frames, extra_args=None, dry_run=False):
    """Run a single experiment."""
    global _stop_requested
    if _stop_requested:
        return False

    exp_name = f"{task}_{env_type}_{method_name}_s{seed}"

    # Check if already completed
    if find_existing_eval(task, method_name, env_type, seed):
        print(f"  [SKIP] {exp_name} (already exists)")
        return True

    # Use num_workers=0 to avoid Windows CreateProcess path-length issues
    # Only pass non-default overrides to keep Hydra dir name short
    cmd = [
        PYTHON, "train.py",
        f"task@_global_={task}",
        f"seed={seed}",
        f"num_train_frames={frames}",
        f"experiment={exp_name}",
        "save_video=false",
        "replay_buffer_num_workers=0",
        "save_snapshot=true",
    ]

    if use_cons:
        cmd.append("use_consistency=true")
    if use_cont:
        cmd.append("use_contrastive=true")
    if env_type == "distractor":
        cmd.append("use_distractors=true")

    if extra_args:
        cmd.extend(extra_args)

    if dry_run:
        print(f"  [DRY] {exp_name}")
        print(f"        {' '.join(cmd)}")
        return True

    print(f"  [RUN] {exp_name} ({frames} frames)")
    start = time.time()

    try:
        result = subprocess.run(
            cmd, cwd=str(PROJECT_DIR),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, timeout=None
        )
        elapsed = time.time() - start
        elapsed_str = str(timedelta(seconds=int(elapsed)))

        # Print last few lines of output
        lines = result.stdout.strip().split('\n')
        for line in lines[-3:]:
            print(f"        {line}")

        if result.returncode == 0:
            print(f"  [OK]  {exp_name} ({elapsed_str})")
            log_progress(exp_name, "OK", elapsed_str)
            return True
        else:
            print(f"  [FAIL] {exp_name} (exit code {result.returncode})")
            log_progress(exp_name, "FAIL", elapsed_str)
            return False

    except Exception as e:
        elapsed = time.time() - start
        print(f"  [ERROR] {exp_name}: {e}")
        log_progress(exp_name, "ERROR", str(e))
        return False


def log_progress(exp_name, status, detail):
    """Append to progress log."""
    log_path = PROJECT_DIR / "exp_local" / "run_all_log.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"{ts} | {status:5s} | {exp_name} | {detail}\n")


def main():
    parser = argparse.ArgumentParser(description="Run all project experiments")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--clean-only", action="store_true")
    parser.add_argument("--distractor-only", action="store_true")
    parser.add_argument("--alpha-only", action="store_true")
    parser.add_argument("--skip-distractor", action="store_true")
    parser.add_argument("--skip-alpha", action="store_true")
    args = parser.parse_args()

    run_clean = not args.distractor_only and not args.alpha_only
    run_distractor = not args.clean_only and not args.alpha_only and not args.skip_distractor
    run_alpha = not args.clean_only and not args.distractor_only and not args.skip_alpha

    # Count total
    total = 0
    if run_clean:
        total += len(TASKS) * len(METHODS) * len(SEEDS)
    if run_distractor:
        total += len(TASKS) * len(METHODS) * len(SEEDS)
    if run_alpha:
        total += 2 * len(ALPHAS) * len(SEEDS)

    print("=" * 60)
    print("  DrQ-v2 Full Experiment Suite")
    print("=" * 60)
    print(f"  Tasks:     {TASKS}")
    print(f"  Methods:   baseline, modA, modB, modAB")
    print(f"  Seeds:     {SEEDS}")
    print(f"  Clean:     {'YES' if run_clean else 'NO'}")
    print(f"  Distractor:{'YES' if run_distractor else 'NO'}")
    print(f"  Alpha:     {'YES' if run_alpha else 'NO'}")
    print(f"  Total runs:{total}")
    print(f"  Project:   {PROJECT_DIR}")
    print("=" * 60)
    print()

    completed = 0
    failed = 0
    skipped = 0
    overall_start = time.time()

    # ---- Phase 1: Clean experiments ----
    if run_clean:
        print("=" * 60)
        print("  Phase 1: Clean Environment Experiments")
        print("=" * 60)
        for task in TASKS:
            frames = TASK_FRAMES[task]
            print(f"\n--- {task} ({frames} frames) ---")
            for method_name, use_cons, use_cont in METHODS:
                for seed in SEEDS:
                    if _stop_requested:
                        break
                    ok = run_experiment(task, method_name, use_cons, use_cont,
                                       seed, "clean", frames, dry_run=args.dry_run)
                    if ok:
                        completed += 1
                    else:
                        failed += 1
                if _stop_requested:
                    break
            if _stop_requested:
                break

    # ---- Phase 2: Distractor experiments ----
    if run_distractor and not _stop_requested:
        print("\n" + "=" * 60)
        print("  Phase 2: Distractor Environment Experiments")
        print("=" * 60)
        for task in TASKS:
            frames = TASK_FRAMES[task]
            print(f"\n--- {task} distractor ({frames} frames) ---")
            for method_name, use_cons, use_cont in METHODS:
                for seed in SEEDS:
                    if _stop_requested:
                        break
                    ok = run_experiment(task, method_name, use_cons, use_cont,
                                       seed, "distractor", frames,
                                       dry_run=args.dry_run)
                    if ok:
                        completed += 1
                    else:
                        failed += 1
                if _stop_requested:
                    break
            if _stop_requested:
                break

    # ---- Phase 3: Alpha sensitivity sweep ----
    if run_alpha and not _stop_requested:
        print("\n" + "=" * 60)
        print("  Phase 3: Alpha Sensitivity Sweep")
        print("=" * 60)
        task = "cartpole_swingup"
        frames = TASK_FRAMES[task]

        # Mod A sweep
        print(f"\n--- Mod A alpha sweep ({task}) ---")
        for alpha in ALPHAS:
            for seed in SEEDS:
                if _stop_requested:
                    break
                exp_name_override = f"alpha_sweep_modA_a{alpha}_s{seed}"
                ok = run_experiment(
                    task, "modA", True, False, seed, "clean", frames,
                    extra_args=[f"consistency_alpha={alpha}",
                                f"experiment={exp_name_override}"],
                    dry_run=args.dry_run)
                if ok:
                    completed += 1
                else:
                    failed += 1
            if _stop_requested:
                break

        # Mod B sweep
        if not _stop_requested:
            print(f"\n--- Mod B alpha sweep ({task}) ---")
            for alpha in ALPHAS:
                for seed in SEEDS:
                    if _stop_requested:
                        break
                    exp_name_override = f"alpha_sweep_modB_a{alpha}_s{seed}"
                    ok = run_experiment(
                        task, "modB", False, True, seed, "clean", frames,
                        extra_args=[f"contrastive_alpha={alpha}",
                                    f"experiment={exp_name_override}"],
                        dry_run=args.dry_run)
                    if ok:
                        completed += 1
                    else:
                        failed += 1
                if _stop_requested:
                    break

    # ---- Summary ----
    total_time = time.time() - overall_start
    total_str = str(timedelta(seconds=int(total_time)))

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Completed: {completed}")
    print(f"  Failed:    {failed}")
    print(f"  Total time:{total_str}")
    if _stop_requested:
        print("  (Stopped early by user)")
    print("=" * 60)

    # ---- Generate plots ----
    if not args.dry_run and not _stop_requested:
        print("\nGenerating result plots...")
        subprocess.run([PYTHON, "scripts/plot_experiment_results.py"],
                       cwd=str(PROJECT_DIR))
        print("Done!")


if __name__ == "__main__":
    main()
