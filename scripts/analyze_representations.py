"""
Analyze encoder representations: effective rank, CKA, t-SNE.

Reads .npz files from eval_representations.py and generates:
  - figures/effective_rank.png
  - figures/cka_clean_vs_distractor.png
  - figures/tsne_latent_{method}.png (one per method)

Usage:
  python scripts/analyze_representations.py --data_dir analysis_data --output_dir figures
"""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ---- Metrics ----

def effective_rank(X):
    """Effective rank via Shannon entropy of normalized singular values.

    erank(X) = exp(H(p)) where p_i = sigma_i / sum(sigma)
    """
    _, S, _ = np.linalg.svd(X, full_matrices=False)
    S = S[S > 1e-10]
    p = S / S.sum()
    entropy = -(p * np.log(p)).sum()
    return np.exp(entropy)


def linear_cka(X, Y, max_samples=2000):
    """Linear Centered Kernel Alignment between two representation matrices.

    X: (n, d1), Y: (n, d2)
    CKA = trace(K_X K_Y) / (||K_X||_F * ||K_Y||_F)
    where K_X = X X^T, K_Y = Y Y^T (kernel formulation).
    """
    # Subsample if needed
    n = min(len(X), len(Y))
    if n > max_samples:
        idx = np.random.RandomState(42).choice(n, max_samples, replace=False)
        X, Y = X[idx], Y[idx]
        n = max_samples

    # Center columns
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)

    # Kernel matrices (n x n)
    Kx = X @ X.T
    Ky = Y @ Y.T

    # CKA = trace(Kx Ky) / (||Kx||_F * ||Ky||_F)
    numer = np.trace(Kx @ Ky)
    denom = np.linalg.norm(Kx, 'fro') * np.linalg.norm(Ky, 'fro')

    return numer / (denom + 1e-10)


# ---- Plotting ----

METHOD_COLORS = {
    'baseline': '#3498db',
    'modA': '#e74c3c',
    'modB': '#2ecc71',
    'modAB': '#9b59b6',
}
METHOD_LABELS = {
    'baseline': 'DrQ-v2 (baseline)',
    'modA': '+ Consistency (Mod A)',
    'modB': '+ Contrastive (Mod B)',
    'modAB': '+ Both (Mod A+B)',
}
METHOD_ORDER = ['baseline', 'modA', 'modB', 'modAB']


def load_all_data(data_dir):
    """Load all .npz files and organize by (task, method, env_type, seed)."""
    data = {}
    data_dir = Path(data_dir)
    for npz_path in sorted(data_dir.glob('*.npz')):
        d = dict(np.load(npz_path, allow_pickle=True))
        task = str(d.get('task_name', 'unknown'))
        method = str(d.get('method', 'unknown'))
        env_type = str(d.get('env_type', 'clean'))
        seed = int(d.get('seed', 0))
        key = (task, method, env_type, seed)
        data[key] = d
    return data


def plot_effective_rank(data, output_dir):
    """Bar chart comparing effective rank across methods per task."""
    tasks = sorted(set(t for t, _, _, _ in data.keys()))
    methods_present = sorted(set(m for _, m, _, _ in data.keys()),
                              key=lambda x: METHOD_ORDER.index(x)
                              if x in METHOD_ORDER else 99)

    fig, axes = plt.subplots(1, len(tasks), figsize=(5 * len(tasks), 5))
    if len(tasks) == 1:
        axes = [axes]

    for ax, task in zip(axes, tasks):
        ranks = defaultdict(list)
        for (t, m, e, s), d in data.items():
            if t == task and e == 'clean':
                r = effective_rank(d['representations'])
                ranks[m].append(r)

        x_pos = []
        x_labels = []
        colors = []
        means = []
        stds = []
        for i, m in enumerate(methods_present):
            if m in ranks:
                x_pos.append(i)
                x_labels.append(METHOD_LABELS.get(m, m))
                colors.append(METHOD_COLORS.get(m, '#888'))
                means.append(np.mean(ranks[m]))
                stds.append(np.std(ranks[m]))

        ax.bar(x_pos, means, yerr=stds, color=colors, capsize=4, alpha=0.8)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, rotation=20, ha='right', fontsize=8)
        ax.set_ylabel('Effective Rank')
        ax.set_title(task.replace('_', ' ').title(), fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Encoder Representation Effective Rank', fontsize=14, y=1.02)
    fig.tight_layout()
    out_path = output_dir / 'effective_rank.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    fig.savefig(str(out_path).replace('.png', '.pdf'), bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_cka_matrix(data, output_dir):
    """CKA similarity between clean and distractor representations."""
    tasks = sorted(set(t for t, _, _, _ in data.keys()))
    methods_present = sorted(set(m for _, m, _, _ in data.keys()),
                              key=lambda x: METHOD_ORDER.index(x)
                              if x in METHOD_ORDER else 99)

    for task in tasks:
        n = len(methods_present)
        cka_matrix = np.zeros((n, n))

        for i, m1 in enumerate(methods_present):
            for j, m2 in enumerate(methods_present):
                cka_vals = []
                for s in range(1, 4):
                    clean_key = (task, m1, 'clean', s)
                    dist_key = (task, m2, 'distractor', s)
                    if clean_key in data and dist_key in data:
                        r1 = data[clean_key]['representations']
                        r2 = data[dist_key]['representations']
                        min_n = min(len(r1), len(r2))
                        if min_n > 10:
                            cka_vals.append(linear_cka(r1[:min_n], r2[:min_n]))
                if cka_vals:
                    cka_matrix[i, j] = np.mean(cka_vals)

        if cka_matrix.sum() == 0:
            continue

        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(cka_matrix, cmap='YlOrRd', vmin=0, vmax=1)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        labels = [METHOD_LABELS.get(m, m) for m in methods_present]
        ax.set_xticklabels([f'{l}\n(distractor)' for l in labels], fontsize=8)
        ax.set_yticklabels([f'{l}\n(clean)' for l in labels], fontsize=8)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f'{cka_matrix[i,j]:.2f}',
                        ha='center', va='center', fontsize=10)
        plt.colorbar(im, ax=ax, label='Linear CKA')
        task_title = task.replace('_', ' ').title()
        ax.set_title(f'CKA: Clean vs Distractor ({task_title})')
        fig.tight_layout()
        out_path = output_dir / f'cka_{task}.png'
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        fig.savefig(str(out_path).replace('.png', '.pdf'), bbox_inches='tight')
        plt.close(fig)
        print(f"Saved {out_path}")


def plot_tsne(data, output_dir):
    """t-SNE visualization of latent space colored by episode return."""
    from sklearn.manifold import TSNE

    # Group by (task, method, env_type)
    groups = defaultdict(list)
    for (t, m, e, s), d in data.items():
        groups[(t, m, e)].append(d)

    for (task, method, env_type), datasets in groups.items():
        # Concatenate across seeds
        all_reprs = np.concatenate([d['representations'] for d in datasets])
        all_returns = np.concatenate([d['episode_returns'] for d in datasets])

        # Subsample if too many points
        n = len(all_reprs)
        if n > 5000:
            idx = np.random.RandomState(42).choice(n, 5000, replace=False)
            all_reprs = all_reprs[idx]
            all_returns = all_returns[idx]

        if n < 10:
            continue

        tsne = TSNE(n_components=2, perplexity=min(30, n // 4),
                     random_state=42, max_iter=1000)
        coords = tsne.fit_transform(all_reprs)

        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(coords[:, 0], coords[:, 1],
                             c=all_returns, cmap='viridis', s=5, alpha=0.6)
        plt.colorbar(scatter, ax=ax, label='Episode Return')
        label = METHOD_LABELS.get(method, method)
        task_title = task.replace('_', ' ').title()
        ax.set_title(f't-SNE: {task_title} - {label} ({env_type})')
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        fig.tight_layout()
        out_path = output_dir / f'tsne_{task}_{method}_{env_type}.png'
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        fig.savefig(str(out_path).replace('.png', '.pdf'), bbox_inches='tight')
        plt.close(fig)
        print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze encoder representations")
    parser.add_argument("--data_dir", type=str, default="analysis_data",
                        help="Directory containing .npz files")
    parser.add_argument("--output_dir", type=str, default="figures",
                        help="Directory to save figures")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.exists():
        print(f"Error: {data_dir} does not exist.")
        print("Run eval_representations.py first.")
        return

    print(f"Loading data from {data_dir}...")
    data = load_all_data(data_dir)
    print(f"Loaded {len(data)} representation files")

    for key in sorted(data.keys()):
        t, m, e, s = key
        n = len(data[key]['representations'])
        print(f"  {t} / {m} / {e} / seed={s}: {n} samples")

    print("\nGenerating analysis plots...")
    plot_effective_rank(data, output_dir)
    plot_cka_matrix(data, output_dir)
    plot_tsne(data, output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
