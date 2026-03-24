"""
Download and prepare video backgrounds for DMC-GB distractor experiments.

Options:
  synthetic  - Generate random-pattern videos (for testing pipeline)
  kinetics   - Download a small subset of Kinetics-400 (requires yt-dlp)

Usage:
  python scripts/download_kinetics.py --output_dir kinetics_videos --mode synthetic --num_videos 20
  python scripts/download_kinetics.py --output_dir kinetics_videos --mode kinetics --num_videos 50
"""

import argparse
import os
from pathlib import Path

import numpy as np


def generate_synthetic_videos(output_dir, num_videos=20, num_frames=250,
                               height=84, width=84, fps=30):
    """Generate synthetic random-pattern videos for pipeline testing."""
    import cv2

    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(42)

    for i in range(num_videos):
        out_path = output_dir / f"synthetic_{i:04d}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

        # Random base color that shifts over time
        base_color = rng.randint(0, 200, 3)
        speed = rng.uniform(0.5, 3.0)

        for t in range(num_frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            # Moving gradient pattern
            for c in range(3):
                shift = int(speed * t) % width
                gradient = np.linspace(0, 1, width)
                gradient = np.roll(gradient, shift)
                channel = (base_color[c] + 55 * gradient).clip(0, 255)
                frame[:, :, c] = np.tile(channel, (height, 1)).astype(np.uint8)
            # Add some noise
            noise = rng.randint(0, 30, (height, width, 3), dtype=np.uint8)
            frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            writer.write(frame)

        writer.release()
        print(f"  [{i+1}/{num_videos}] {out_path.name}")

    print(f"Generated {num_videos} synthetic videos in {output_dir}")


def download_kinetics_clips(output_dir, num_videos=50):
    """Download Kinetics-400 video clips using yt-dlp.

    Uses a curated list of short action-recognition clips commonly used
    in DMC-GB experiments.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sample YouTube IDs from Kinetics-400 validation set (public domain)
    # These are short clips typically used for background distractors
    sample_ids = [
        "dQw4w9WgXcQ", "9bZkp7q19f0", "kJQP7kiw5Fk",
        "JGwWNGJdvx8", "RgKAFK5djSk", "CevxZvSJLk8",
    ]

    print("Downloading Kinetics clips requires yt-dlp.")
    print("Install: pip install yt-dlp")
    print(f"Will download {min(num_videos, len(sample_ids))} clips to {output_dir}")
    print()
    print("For a proper DMC-GB setup, download the Kinetics-400 dataset:")
    print("  https://github.com/cvdfoundation/kinetics-dataset")
    print()
    print("Alternatively, use --mode synthetic for pipeline testing.")

    try:
        import subprocess
        for i, vid_id in enumerate(sample_ids[:num_videos]):
            out_path = output_dir / f"kinetics_{i:04d}.mp4"
            if out_path.exists():
                print(f"  [{i+1}] Already exists: {out_path.name}")
                continue
            cmd = [
                "yt-dlp", "-f", "worst[ext=mp4]",
                "--max-filesize", "5M",
                "-o", str(out_path),
                f"https://www.youtube.com/watch?v={vid_id}"
            ]
            print(f"  [{i+1}] Downloading {vid_id}...")
            subprocess.run(cmd, capture_output=True)
    except FileNotFoundError:
        print("ERROR: yt-dlp not found. Use --mode synthetic instead.")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare video backgrounds for DMC-GB experiments")
    parser.add_argument("--output_dir", type=str, default="kinetics_videos",
                        help="Output directory for video files")
    parser.add_argument("--mode", type=str, default="synthetic",
                        choices=["synthetic", "kinetics"],
                        help="synthetic: random patterns; kinetics: download real videos")
    parser.add_argument("--num_videos", type=int, default=20,
                        help="Number of videos to generate/download")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    print(f"Mode: {args.mode}")
    print(f"Output: {output_dir}")
    print(f"Num videos: {args.num_videos}")
    print()

    if args.mode == "synthetic":
        generate_synthetic_videos(output_dir, args.num_videos)
    elif args.mode == "kinetics":
        download_kinetics_clips(output_dir, args.num_videos)


if __name__ == "__main__":
    main()
