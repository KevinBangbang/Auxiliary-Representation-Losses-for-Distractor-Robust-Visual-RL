"""
Evaluate trained agents and save encoder representations for analysis.

For each experiment directory with a snapshot.pt:
  1. Load the trained agent
  2. Run eval episodes in the environment
  3. Save encoder representations, Q-values, and episode returns

Usage:
  python scripts/eval_representations.py --exp_dir exp_local/2026.03.03/123456_...
  python scripts/eval_representations.py --scan_dir exp_local --output_dir analysis_data
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

import dmc
import utils


def collect_representations(agent, env, num_episodes, device):
    """Run eval episodes and collect encoder representations."""
    all_reprs = []
    all_q_values = []
    all_episode_returns = []
    all_episode_ids = []

    for ep in range(num_episodes):
        time_step = env.reset()
        episode_reward = 0
        step_reprs = []
        step_qs = []

        while not time_step.last():
            obs = torch.as_tensor(time_step.observation, device=device)
            obs = obs.unsqueeze(0).float()

            with torch.no_grad():
                enc_obs = agent.encoder(obs)
                # Use actor trunk to get compact 50-dim features
                trunk_obs = agent.actor.trunk(enc_obs)
                step_reprs.append(trunk_obs.cpu().numpy()[0])

                dist = agent.actor(enc_obs, 0.1)
                action = dist.mean
                q1, q2 = agent.critic(enc_obs, action)
                q = torch.min(q1, q2).item()
                step_qs.append(q)

            action_np = action.cpu().numpy()[0]
            time_step = env.step(action_np)
            episode_reward += time_step.reward

        all_reprs.extend(step_reprs)
        all_q_values.extend(step_qs)
        all_episode_returns.extend([episode_reward] * len(step_reprs))
        all_episode_ids.extend([ep] * len(step_reprs))

    return {
        'representations': np.array(all_reprs),
        'q_values': np.array(all_q_values),
        'episode_returns': np.array(all_episode_returns),
        'episode_ids': np.array(all_episode_ids),
    }


def process_experiment(exp_dir, output_dir, num_episodes=10, device='cuda'):
    """Process a single experiment directory."""
    config_path = exp_dir / '.hydra' / 'config.yaml'
    snapshot_path = exp_dir / 'snapshot.pt'

    if not config_path.exists():
        print(f"  Skipping {exp_dir}: no config.yaml")
        return
    if not snapshot_path.exists():
        print(f"  Skipping {exp_dir}: no snapshot.pt")
        return

    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    task_name = cfg.get('task_name')
    seed = cfg.get('seed', 1)
    use_distractors = cfg.get('use_distractors', False)
    use_consistency = cfg.get('use_consistency', False)
    use_contrastive = cfg.get('use_contrastive', False)

    # Determine method name
    if use_consistency and use_contrastive:
        method = 'modAB'
    elif use_consistency:
        method = 'modA'
    elif use_contrastive:
        method = 'modB'
    else:
        method = 'baseline'

    env_type = 'distractor' if use_distractors else 'clean'
    label = f"{task_name}_{env_type}_{method}_s{seed}"
    print(f"  Processing: {label}")

    # Load agent
    payload = torch.load(snapshot_path, map_location=device, weights_only=False)
    agent = payload['agent']
    agent.train(False)

    # Create eval environment (always clean for representation comparison)
    env = dmc.make(task_name, cfg.get('frame_stack', 3),
                   cfg.get('action_repeat', 2), seed)

    # Collect representations
    data = collect_representations(agent, env, num_episodes, device)
    data['task_name'] = task_name
    data['method'] = method
    data['env_type'] = env_type
    data['seed'] = seed

    # Also collect in distractor env if applicable
    if use_distractors:
        distractor_video_dir = cfg.get('distractor_video_dir', './kinetics_videos')
        env_dist = dmc.make(task_name, cfg.get('frame_stack', 3),
                            cfg.get('action_repeat', 2), seed,
                            use_distractors=True,
                            distractor_video_dir=distractor_video_dir)
        data_dist = collect_representations(agent, env_dist, num_episodes, device)
        data['representations_distractor'] = data_dist['representations']

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{label}.npz"
    np.savez_compressed(out_path, **{k: v for k, v in data.items()
                                     if isinstance(v, np.ndarray)},
                        task_name=task_name, method=method,
                        env_type=env_type, seed=seed)
    print(f"  Saved: {out_path}")


def scan_and_process(scan_dir, output_dir, num_episodes=10, device='cuda'):
    """Scan all experiment directories and process those with snapshots."""
    scan_dir = Path(scan_dir)
    output_dir = Path(output_dir)

    exp_dirs = []
    for date_dir in sorted(scan_dir.iterdir()):
        if not date_dir.is_dir():
            continue
        for exp_dir in sorted(date_dir.iterdir()):
            if not exp_dir.is_dir():
                continue
            if (exp_dir / 'snapshot.pt').exists():
                exp_dirs.append(exp_dir)

    print(f"Found {len(exp_dirs)} experiments with snapshots")
    for exp_dir in exp_dirs:
        process_experiment(exp_dir, output_dir, num_episodes, device)


def main():
    parser = argparse.ArgumentParser(
        description="Collect encoder representations from trained agents")
    parser.add_argument("--exp_dir", type=str, default=None,
                        help="Single experiment directory")
    parser.add_argument("--scan_dir", type=str, default=None,
                        help="Root directory to scan for experiments")
    parser.add_argument("--output_dir", type=str, default="analysis_data",
                        help="Output directory for .npz files")
    parser.add_argument("--num_episodes", type=int, default=10,
                        help="Number of eval episodes per agent")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.exp_dir:
        process_experiment(Path(args.exp_dir), Path(args.output_dir),
                           args.num_episodes, args.device)
    elif args.scan_dir:
        scan_and_process(args.scan_dir, args.output_dir,
                         args.num_episodes, args.device)
    else:
        print("Provide --exp_dir or --scan_dir")


if __name__ == "__main__":
    main()
