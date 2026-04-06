"""
One-script runner for ALL project experiments.

Priority-ordered execution with parallel support (2 experiments at once).

Priority order:
  P0: walker/cheetah distractor s2,s3  (core paper claims)
  P1: cheetah clean s2,s3              (needed for retention calc)
  P2: walker clean remaining           (supplementary)
  P3: cartpole (already done)
  P4: alpha sweep (already done)

Features:
  - Runs 2 experiments in parallel (GPU 24GB, each uses ~2.5GB)
  - Skips experiments whose eval.csv already exists
  - Saves progress log to exp_local/run_all_log.txt
  - Ctrl+C gracefully stops after current experiments finish

Usage:
  python scripts/run_all.py                     # run everything (2 parallel)
  python scripts/run_all.py --parallel 1        # serial mode
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
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
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


def build_cmd(task, method_name, use_cons, use_cont, seed, env_type,
              frames, extra_args=None):
    """Build command list for an experiment."""
    exp_name = f"{task}_{env_type}_{method_name}_s{seed}"
    cmd = [
        PYTHON, "train.py",
        f"task@_global_={task}",
        f"seed={seed}",
        f"num_train_frames={frames}",
        f"experiment={exp_name}",
        "save_video=false",
        "replay_buffer_num_workers=0",
        "save_snapshot=true",
        "replay_buffer_size=100000",
        f"hydra.run.dir=./exp_local/${{now:%Y.%m.%d}}/{exp_name}",
    ]
    if use_cons:
        cmd.append("use_consistency=true")
    if use_cont:
        cmd.append("use_contrastive=true")
    if env_type == "distractor":
        cmd.append("use_distractors=true")
    if extra_args:
        cmd.extend(extra_args)
    return exp_name, cmd


def run_experiment(task, method_name, use_cons, use_cont, seed, env_type,
                   frames, extra_args=None, dry_run=False):
    """Run a single experiment. Returns (exp_name, status)."""
    global _stop_requested
    exp_name = f"{task}_{env_type}_{method_name}_s{seed}"

    if _stop_requested:
        return exp_name, "STOP"

    # Check if already completed
    if find_existing_eval(task, method_name, env_type, seed):
        print(f"  [SKIP] {exp_name}")
        return exp_name, "SKIP"

    exp_name, cmd = build_cmd(task, method_name, use_cons, use_cont, seed,
                              env_type, frames, extra_args)

    if dry_run:
        print(f"  [DRY] {exp_name}")
        print(f"        {' '.join(cmd)}")
        return exp_name, "DRY"

    print(f"  [RUN]  {exp_name} ({frames} frames)")
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
            print(f"  [OK]   {exp_name} ({elapsed_str})")
            log_progress(exp_name, "OK", elapsed_str)
            return exp_name, "OK"
        else:
            print(f"  [FAIL] {exp_name} (exit code {result.returncode}, {elapsed_str})")
            log_progress(exp_name, "FAIL", elapsed_str)
            return exp_name, "FAIL"

    except Exception as e:
        elapsed = time.time() - start
        print(f"  [ERROR] {exp_name}: {e}")
        log_progress(exp_name, "ERROR", str(e))
        return exp_name, "ERROR"


def run_pair(pair, dry_run=False):
    """Run 2 experiments in parallel using threads, or 1 if pair has one entry."""
    results = []

    def _worker(args):
        return run_experiment(*args, dry_run=dry_run)

    if len(pair) == 1:
        results.append(_worker(pair[0]))
    else:
        threads = []
        thread_results = [None, None]

        def _thread_fn(idx, args):
            thread_results[idx] = _worker(args)

        for i, args in enumerate(pair):
            t = threading.Thread(target=_thread_fn, args=(i, args))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        results = [r for r in thread_results if r is not None]

    return results


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
    parser.add_argument("--parallel", type=int, default=1,
                        help="Number of experiments to run in parallel (default: 1)")
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

    n_parallel = args.parallel
    print("=" * 60)
    print("  DrQ-v2 Full Experiment Suite")
    print("=" * 60)
    print(f"  Tasks:     {TASKS}")
    print(f"  Methods:   baseline, modA, modB, modAB")
    print(f"  Seeds:     {SEEDS}")
    print(f"  Parallel:  {n_parallel}")
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

    # ---- Build priority queue ----
    # Order by paper importance:
    #   P0: walker/cheetah distractor s2,s3  (core paper claims)
    #   P1: cheetah clean s2,s3              (needed for retention calc)
    #   P2: walker clean remaining           (supplementary)
    #   P3: cartpole (already done)
    #   P4: alpha sweep                      (already done, will skip)
    #
    # Within each priority: baseline & modA first (stable),
    # modB & modAB last (slow / collapse-prone)

    METHODS_STABLE_FIRST = [
        ("baseline", False, False),
        ("modA", True, False),
        ("modAB", True, True),
        ("modB", False, True),   # most fragile, run last
    ]

    priority_queue = []

    if run_distractor or run_clean:
        # P0: distractor experiments for walker & cheetah (paper core)
        if run_distractor:
            for task in ["walker_walk", "cheetah_run"]:
                for m_name, use_cons, use_cont in METHODS_STABLE_FIRST:
                    for seed in SEEDS:
                        priority_queue.append(
                            (0, task, m_name, use_cons, use_cont, seed,
                             "distractor", TASK_FRAMES[task], None))

        # P1: clean experiments for cheetah (needed for retention %)
        if run_clean:
            for m_name, use_cons, use_cont in METHODS_STABLE_FIRST:
                for seed in SEEDS:
                    priority_queue.append(
                        (1, "cheetah_run", m_name, use_cons, use_cont, seed,
                         "clean", TASK_FRAMES["cheetah_run"], None))

        # P2: remaining walker clean
        if run_clean:
            for m_name, use_cons, use_cont in METHODS_STABLE_FIRST:
                for seed in SEEDS:
                    priority_queue.append(
                        (2, "walker_walk", m_name, use_cons, use_cont, seed,
                         "clean", TASK_FRAMES["walker_walk"], None))

        # P3: cartpole (all seeds already done, will auto-skip)
        for env_type in (["clean"] if run_clean else []) + \
                        (["distractor"] if run_distractor else []):
            for m_name, use_cons, use_cont in METHODS_STABLE_FIRST:
                for seed in SEEDS:
                    priority_queue.append(
                        (3, "cartpole_swingup", m_name, use_cons, use_cont,
                         seed, env_type, TASK_FRAMES["cartpole_swingup"], None))

    # P4: alpha sweep (already done, will auto-skip)
    if run_alpha:
        task = "cartpole_swingup"
        frames = TASK_FRAMES[task]
        for alpha in ALPHAS:
            for seed in SEEDS:
                priority_queue.append(
                    (4, task, "modA", True, False, seed, "clean", frames,
                     [f"consistency_alpha={alpha}",
                      f"experiment=alpha_sweep_modA_a{alpha}_s{seed}"]))
                priority_queue.append(
                    (4, task, "modB", False, True, seed, "clean", frames,
                     [f"contrastive_alpha={alpha}",
                      f"experiment=alpha_sweep_modB_a{alpha}_s{seed}"]))

    # Sort by priority (stable sort preserves inner order)
    priority_queue.sort(key=lambda x: x[0])

    priority_names = {0: "P0-distractor(core)", 1: "P1-cheetah-clean",
                      2: "P2-walker-clean", 3: "P3-cartpole(done)",
                      4: "P4-alpha(done)"}

    # ---- Pre-filter: skip already-completed, collect runnable ----
    runnable = []
    for entry in priority_queue:
        prio, task, m_name, use_cons, use_cont, seed, env_type, frames, extra = entry
        exp_name = f"{task}_{env_type}_{m_name}_s{seed}"
        if find_existing_eval(task, m_name, env_type, seed):
            print(f"  [SKIP] {exp_name}")
            skipped += 1
        else:
            runnable.append(entry)

    print(f"\n  Skipped {skipped} already-completed experiments")
    print(f"  Remaining: {len(runnable)} experiments to run")
    print(f"  Running {n_parallel} at a time\n")

    # ---- Execute in parallel batches ----
    i = 0
    while i < len(runnable) and not _stop_requested:
        batch = runnable[i:i + n_parallel]
        i += n_parallel

        # Print batch info
        prio = batch[0][0]
        batch_names = []
        for entry in batch:
            p, task, m_name, use_cons, use_cont, seed, env_type, frames, extra = entry
            batch_names.append(f"{task}_{env_type}_{m_name}_s{seed}")

        print(f"\n--- [{priority_names.get(prio, f'P{prio}')}] "
              f"Batch: {' + '.join(batch_names)} ---")

        # Build argument tuples for run_pair
        pair_args = []
        for entry in batch:
            _, task, m_name, use_cons, use_cont, seed, env_type, frames, extra = entry
            pair_args.append((task, m_name, use_cons, use_cont, seed,
                              env_type, frames, extra))

        results = run_pair(pair_args, dry_run=args.dry_run)

        for exp_name, status in results:
            if status in ("OK", "SKIP", "DRY"):
                completed += 1
            else:
                failed += 1

    # ---- Summary ----
    total_time = time.time() - overall_start
    total_str = str(timedelta(seconds=int(total_time)))

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Completed: {completed}")
    print(f"  Skipped:   {skipped}")
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
