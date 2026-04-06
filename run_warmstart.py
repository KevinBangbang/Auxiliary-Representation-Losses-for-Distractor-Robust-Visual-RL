"""
Run warm-start Mod B experiment: cartpole clean seed=2.
Launch with: python run_warmstart.py
Progress: tail -f warmstart_progress.log
"""
import subprocess
import sys
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

cmd = [
    sys.executable, "train.py",
    "task@_global_=cartpole_swingup",
    "num_train_frames=500000",
    "seed=2",
    "use_contrastive=true",
    "contrastive_warmstart_steps=100000",
    "experiment=warmstart_modB_100k_s2",
    "replay_buffer_num_workers=0",
    "save_video=false",
    "save_snapshot=true",
    "replay_buffer_size=200000",
]

print("Starting warm-start experiment...")
print("Command:", " ".join(cmd))
print("Monitor: tail -f warmstart_progress.log")
print("Or check: eval.csv in the exp_local/2026.04.06/ directory")

with open("warmstart_progress.log", "w") as logf:
    proc = subprocess.Popen(
        cmd,
        stdout=logf,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    print(f"PID: {proc.pid}")
    print("Running in foreground. Ctrl+C to stop.")
    try:
        ret = proc.wait()
        print(f"Done! Exit code: {ret}")
    except KeyboardInterrupt:
        proc.terminate()
        print("Stopped by user.")
