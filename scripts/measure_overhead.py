"""
Measure computational overhead of each method variant.

Runs a fixed number of update steps with synthetic batches and reports
wall-clock time per step for each configuration.

Usage:
  python scripts/measure_overhead.py
  python scripts/measure_overhead.py --num_steps 200 --batch_size 256
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from drqv2 import DrQV2Agent


def create_agent(obs_shape, action_shape, device, use_consistency=False,
                 use_contrastive=False):
    """Create a DrQV2Agent with the given auxiliary loss configuration."""
    return DrQV2Agent(
        obs_shape=obs_shape,
        action_shape=action_shape,
        device=device,
        lr=1e-4,
        feature_dim=50,
        hidden_dim=1024,
        critic_target_tau=0.01,
        num_expl_steps=0,
        update_every_steps=1,
        stddev_schedule='0.2',
        stddev_clip=0.3,
        use_tb=False,
        use_consistency=use_consistency,
        consistency_alpha=0.1,
        use_contrastive=use_contrastive,
        contrastive_alpha=0.1,
        contrastive_tau=0.1,
        contrastive_epsilon=5.0,
    )


def fake_replay_iter(batch_size, obs_shape, action_dim, device):
    """Generate fake replay batches."""
    while True:
        obs = np.random.randint(0, 256, (batch_size, *obs_shape), dtype=np.uint8)
        action = np.random.randn(batch_size, action_dim).astype(np.float32)
        reward = np.random.randn(batch_size, 1).astype(np.float32)
        discount = np.ones((batch_size, 1), dtype=np.float32)
        next_obs = np.random.randint(0, 256, (batch_size, *obs_shape), dtype=np.uint8)
        yield (obs, action, reward, discount, next_obs)


def measure_method(name, agent, replay_iter, num_warmup=20, num_steps=100):
    """Measure average time per update step."""
    # Warmup
    for i in range(num_warmup):
        agent.update(replay_iter, step=i)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.time()
    for i in range(num_steps):
        agent.update(replay_iter, step=num_warmup + i)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.time() - start
    ms_per_step = (elapsed / num_steps) * 1000

    print(f"  {name:30s}: {ms_per_step:8.2f} ms/step  "
          f"({elapsed:.1f}s for {num_steps} steps)")
    return ms_per_step


def main():
    parser = argparse.ArgumentParser(
        description="Measure computational overhead of method variants")
    parser.add_argument("--num_steps", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'

    obs_shape = (9, 84, 84)  # 3 frames x 3 channels
    action_shape = (6,)      # walker_walk action dim

    print("============================================")
    print("  Computational Overhead Benchmark")
    print("============================================")
    print(f"  Device:     {device}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Steps:      {args.num_steps}")
    print(f"  Obs shape:  {obs_shape}")
    print(f"  Action dim: {action_shape}")
    print("============================================")
    print()

    methods = [
        ("Baseline (DrQ-v2)", False, False),
        ("+ Mod A (Consistency)", True, False),
        ("+ Mod B (Contrastive)", False, True),
        ("+ Mod A + B (Both)", True, True),
    ]

    results = {}
    for name, use_cons, use_cont in methods:
        agent = create_agent(obs_shape, action_shape, device,
                             use_consistency=use_cons,
                             use_contrastive=use_cont)
        replay = fake_replay_iter(args.batch_size, obs_shape,
                                  action_shape[0], device)
        ms = measure_method(name, agent, replay,
                            num_steps=args.num_steps)
        results[name] = ms

    # Print summary
    baseline_ms = results["Baseline (DrQ-v2)"]
    print()
    print("  Overhead relative to baseline:")
    for name, ms in results.items():
        overhead = ((ms - baseline_ms) / baseline_ms) * 100
        print(f"    {name:30s}: +{overhead:5.1f}%")

    # Save results
    output_dir = Path("figures")
    output_dir.mkdir(exist_ok=True)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    names = list(results.keys())
    times = list(results.values())
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(names)), times, color=colors[:len(names)], alpha=0.8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=15, ha='right', fontsize=9)
    ax.set_ylabel('Time per update step (ms)')
    ax.set_title('Computational Overhead Comparison')
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, val in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{val:.1f}ms', ha='center', va='bottom', fontsize=9)

    fig.tight_layout()
    out_path = output_dir / 'computational_overhead.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
