"""
Overnight experiment runner. Just launch and forget:
  python run_overnight.py

Runs walker distractor experiments (baseline & modA, seeds 2,3) sequentially.
Each takes ~4h, total ~16h. Progress saved to overnight_log.txt.
"""
import subprocess
import sys
import os
from datetime import datetime

os.chdir(os.path.dirname(os.path.abspath(__file__)))
PYTHON = os.path.join("venv", "Scripts", "python.exe")
LOG = "overnight_log.txt"

experiments = [
    ("walker_walk_distractor_baseline_s2", {
        "task@_global_": "walker_walk",
        "use_distractors": "true",
        "seed": "2",
    }),
    ("walker_walk_distractor_baseline_s3", {
        "task@_global_": "walker_walk",
        "use_distractors": "true",
        "seed": "3",
    }),
    ("walker_walk_distractor_modA_s2", {
        "task@_global_": "walker_walk",
        "use_distractors": "true",
        "use_consistency": "true",
        "seed": "2",
    }),
    ("walker_walk_distractor_modA_s3", {
        "task@_global_": "walker_walk",
        "use_distractors": "true",
        "use_consistency": "true",
        "seed": "3",
    }),
]

def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line)
    with open(LOG, "a") as f:
        f.write(line + "\n")

log("=== Overnight runner started ===")
log(f"Total experiments: {len(experiments)}")

for i, (name, params) in enumerate(experiments):
    cmd = [PYTHON, "train.py"]
    cmd.append(f"task@_global_={params.pop('task@_global_')}")
    cmd.append("num_train_frames=1000000")
    cmd.append("replay_buffer_num_workers=0")
    cmd.append("replay_buffer_size=200000")
    cmd.append("save_video=false")
    cmd.append("save_snapshot=true")
    cmd.append(f"experiment={name}")
    for k, v in params.items():
        cmd.append(f"{k}={v}")

    log(f"[{i+1}/{len(experiments)}] Starting {name}")
    try:
        result = subprocess.run(cmd, timeout=6*3600)
        log(f"[{i+1}/{len(experiments)}] {name} finished (exit={result.returncode})")
    except subprocess.TimeoutExpired:
        log(f"[{i+1}/{len(experiments)}] {name} TIMEOUT (6h limit)")
    except Exception as e:
        log(f"[{i+1}/{len(experiments)}] {name} ERROR: {e}")

log("=== All done ===")
