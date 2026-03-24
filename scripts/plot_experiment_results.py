"""
Plot experiment results for the Auxiliary Representation Losses project.

Uses an explicit directory manifest to avoid picking up incomplete/failed runs.
Generates 6 publication-quality figures in figures/.

Usage:
  python scripts/plot_experiment_results.py
  python scripts/plot_experiment_results.py --output_dir figures
"""

import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ---- Style ----

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'legend.fontsize': 9,
    'figure.dpi': 150,
})

METHOD_COLORS  = {'baseline': '#3498db', 'modA': '#e74c3c',
                  'modB': '#2ecc71',    'modAB': '#9b59b6'}
METHOD_LABELS  = {'baseline': 'DrQ-v2 (baseline)',
                  'modA':     '+ Consistency (Mod A)',
                  'modB':     '+ Contrastive (Mod B)',
                  'modAB':    '+ Both (Mod A+B)'}
METHOD_ORDER   = ['baseline', 'modA', 'modB', 'modAB']
ALPHA_COLORS   = {0.01: '#e74c3c', 0.1: '#2ecc71', 0.5: '#3498db', 1.0: '#9b59b6'}

# ---- Explicit manifest of correct experiment directories ----
# Only these directories are used; early failed/incomplete runs are ignored.

BASE = Path('exp_local')

# Experiments with known fixed paths (run_core.sh and run_distractor_s23)
CORE_EXPERIMENTS_FIXED = {
    # Clean env — seed=1
    ('clean', 'baseline', 1): BASE / '2026.03.03/cartpole_swingup_clean_baseline_s1',
    ('clean', 'modA',     1): BASE / '2026.03.03/cartpole_swingup_clean_modA_s1',
    ('clean', 'modB',     1): BASE / '2026.03.03/cartpole_swingup_clean_modB_s1',
    ('clean', 'modAB',    1): BASE / '2026.03.03/cartpole_swingup_clean_modAB_s1',
    # Distractor env — seeds 1, 2, 3
    ('distractor', 'baseline', 1): BASE / '2026.03.03/cartpole_swingup_distractor_baseline_s1',
    ('distractor', 'modA',     1): BASE / '2026.03.04/cartpole_swingup_distractor_modA_s1',
    ('distractor', 'modB',     1): BASE / '2026.03.04/cartpole_swingup_distractor_modB_s1',
    ('distractor', 'modAB',    1): BASE / '2026.03.04/cartpole_swingup_distractor_modAB_s1',
    ('distractor', 'baseline', 2): BASE / '2026.03.05/cartpole_swingup_distractor_baseline_s2',
    ('distractor', 'modA',     2): BASE / '2026.03.05/cartpole_swingup_distractor_modA_s2',
    ('distractor', 'modB',     2): BASE / '2026.03.06/cartpole_swingup_distractor_modB_s2',
    ('distractor', 'modAB',    2): BASE / '2026.03.06/cartpole_swingup_distractor_modAB_s2',
    ('distractor', 'baseline', 3): BASE / '2026.03.05/cartpole_swingup_distractor_baseline_s3',
    ('distractor', 'modA',     3): BASE / '2026.03.05/cartpole_swingup_distractor_modA_s3',
    ('distractor', 'modB',     3): BASE / '2026.03.06/cartpole_swingup_distractor_modB_s3',
    ('distractor', 'modAB',    3): BASE / '2026.03.06/cartpole_swingup_distractor_modAB_s3',
}

# Clean seeds 2,3 — date varies, so search by directory name
CLEAN_S23_NAMES = {
    ('clean', 'baseline', 2): 'cartpole_swingup_clean_baseline_s2',
    ('clean', 'modA',     2): 'cartpole_swingup_clean_modA_s2',
    ('clean', 'modB',     2): 'cartpole_swingup_clean_modB_s2',
    ('clean', 'modAB',    2): 'cartpole_swingup_clean_modAB_s2',
    ('clean', 'baseline', 3): 'cartpole_swingup_clean_baseline_s3',
    ('clean', 'modA',     3): 'cartpole_swingup_clean_modA_s3',
    ('clean', 'modB',     3): 'cartpole_swingup_clean_modB_s3',
    ('clean', 'modAB',    3): 'cartpole_swingup_clean_modAB_s3',
}


def find_by_name(name: str) -> Path | None:
    """Search exp_local/<date>/<name> for the latest matching directory."""
    matches = sorted(BASE.glob(f'*/{name}'))
    return matches[-1] if matches else None


def build_core_experiments() -> dict:
    """Combine fixed paths with auto-discovered clean s2/s3 paths."""
    experiments = dict(CORE_EXPERIMENTS_FIXED)
    for key, name in CLEAN_S23_NAMES.items():
        p = find_by_name(name)
        if p:
            experiments[key] = p
    return experiments


CORE_EXPERIMENTS = build_core_experiments()

ALPHA_EXPERIMENTS = {
    # (mod, alpha) -> relative path
    ('modA', 0.01): BASE / '2026.03.04/alpha_modA_a0.01_s1',
    ('modA', 0.1):  BASE / '2026.03.04/alpha_modA_a0.1_s1',
    ('modA', 0.5):  BASE / '2026.03.04/alpha_modA_a0.5_s1',
    ('modA', 1.0):  BASE / '2026.03.04/alpha_modA_a1.0_s1',
    ('modB', 0.01): BASE / '2026.03.05/alpha_modB_a0.01_s1',
    ('modB', 0.1):  BASE / '2026.03.05/alpha_modB_a0.1_s1',
    ('modB', 0.5):  BASE / '2026.03.05/alpha_modB_a0.5_s1',
    ('modB', 1.0):  BASE / '2026.03.05/alpha_modB_a1.0_s1',
}

MIN_FRAMES = 480_000   # require at least this many frames to count as complete


# ---- Data loading ----

def load_eval(path: Path) -> pd.DataFrame | None:
    """Load eval.csv; return None if missing or incomplete."""
    csv = path / 'eval.csv'
    if not csv.exists():
        return None
    df = pd.read_csv(csv)
    if df.empty or df['frame'].max() < MIN_FRAMES:
        return None
    return df


def load_core_data() -> dict:
    """Return {(env, method, seed): DataFrame}."""
    data = {}
    for key, d in CORE_EXPERIMENTS.items():
        df = load_eval(d)
        if df is not None:
            data[key] = df
        else:
            print(f'  [skip] {d.name}')
    return data


def load_alpha_data() -> dict:
    """Return {(mod, alpha): DataFrame}."""
    data = {}
    for key, d in ALPHA_EXPERIMENTS.items():
        df = load_eval(d)
        if df is not None:
            data[key] = df
        else:
            print(f'  [skip alpha] {d.name}')
    return data


def aggregate_seeds(dfs: list[pd.DataFrame]):
    """Align on common frames and compute mean ± std across seeds."""
    frames = dfs[0]['frame'].values
    for df in dfs[1:]:
        frames = np.intersect1d(frames, df['frame'].values)
    if len(frames) == 0:
        return None, None, None
    rewards = np.array([df.set_index('frame').loc[frames, 'episode_reward'].values
                        for df in dfs])
    return frames, rewards.mean(0), rewards.std(0)


def final_reward(df: pd.DataFrame) -> float:
    """Mean of last 5 eval points for stability."""
    return df['episode_reward'].iloc[-5:].mean()


# ---- Helpers ----

def fmt_M(x, _):
    return f'{x/1e6:.2f}M' if x > 0 else '0'


def save(fig, path: Path):
    # Save both PNG (for preview) and PDF (for LaTeX)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    pdf_path = path.with_suffix('.pdf')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {path} + {pdf_path}')


# ======================================================================
# Figure 1 — Distractor learning curves (main result, 3 seeds ± std)
# ======================================================================

def fig_distractor_curves(data: dict, out: Path):
    """Learning curves in distractor env, shaded ±1σ over 3 seeds."""
    fig, ax = plt.subplots(figsize=(7, 5))

    for method in METHOD_ORDER:
        dfs = [data[('distractor', method, s)]
               for s in [1, 2, 3]
               if ('distractor', method, s) in data]
        if not dfs:
            continue
        frames, mean, std = aggregate_seeds(dfs)
        if frames is None:
            continue
        color = METHOD_COLORS[method]
        label = METHOD_LABELS[method]
        lw = 2.5 if method == 'baseline' else 2.0
        ax.plot(frames, mean, color=color, lw=lw, label=label)
        ax.fill_between(frames, mean - std, mean + std, color=color, alpha=0.18)

    ax.set_title('Cartpole Swingup — Distractor Environment', fontsize=13)
    ax.set_xlabel('Environment Frames')
    ax.set_ylabel('Episode Reward')
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_M))
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save(fig, out / 'fig1_distractor_curves.png')


# ======================================================================
# Figure 2 — Clean learning curves (seed=1, no shading)
# ======================================================================

def fig_clean_curves(data: dict, out: Path):
    """Learning curves in clean env (seed=1)."""
    fig, ax = plt.subplots(figsize=(7, 5))

    for method in METHOD_ORDER:
        if ('clean', method, 1) not in data:
            continue
        df = data[('clean', method, 1)]
        color = METHOD_COLORS[method]
        label = METHOD_LABELS[method]
        lw = 2.5 if method == 'baseline' else 2.0
        ax.plot(df['frame'], df['episode_reward'], color=color, lw=lw, label=label)

    ax.set_title('Cartpole Swingup — Clean Environment (seed=1)', fontsize=13)
    ax.set_xlabel('Environment Frames')
    ax.set_ylabel('Episode Reward')
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_M))
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save(fig, out / 'fig2_clean_curves.png')


# ======================================================================
# Figure 3 — Final performance bar chart (clean vs distractor)
# ======================================================================

def fig_final_performance(data: dict, out: Path):
    """Grouped bar: clean (seed=1) and distractor (mean±std, 3 seeds)."""
    methods = METHOD_ORDER
    x = np.arange(len(methods))
    width = 0.35

    clean_means, clean_stds = [], []
    dist_means,  dist_stds  = [], []

    for m in methods:
        # Clean — seed=1 only
        r_c = final_reward(data[('clean', m, 1)]) if ('clean', m, 1) in data else 0
        clean_means.append(r_c)
        clean_stds.append(0)

        # Distractor — mean over available seeds
        vals = [final_reward(data[('distractor', m, s)])
                for s in [1, 2, 3] if ('distractor', m, s) in data]
        dist_means.append(np.mean(vals) if vals else 0)
        dist_stds.append(np.std(vals) if len(vals) > 1 else 0)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [METHOD_COLORS[m] for m in methods]

    bars_c = ax.bar(x - width/2, clean_means, width,
                    color=colors, alpha=0.85, label='Clean (seed=1)',
                    yerr=clean_stds, capsize=4, error_kw={'elinewidth': 1.5})
    bars_d = ax.bar(x + width/2, dist_means, width,
                    color=colors, alpha=0.45, label='Distractor (mean±std, 3 seeds)',
                    yerr=dist_stds, capsize=4, error_kw={'elinewidth': 1.5},
                    hatch='//')

    # Value labels on bars
    for bar, v in zip(bars_c, clean_means):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 8,
                    f'{v:.0f}', ha='center', va='bottom', fontsize=8)
    for bar, v in zip(bars_d, dist_means):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 8,
                    f'{v:.0f}', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[m] for m in methods], rotation=15, ha='right')
    ax.set_ylabel('Episode Reward')
    ax.set_title('Final Performance — Cartpole Swingup', fontsize=13)
    ax.legend()
    ax.set_ylim(0, max(clean_means + dist_means) * 1.18)
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    save(fig, out / 'fig3_final_performance.png')


# ======================================================================
# Figure 4 — Robustness bar chart (% retention clean→distractor)
# ======================================================================

def fig_robustness(data: dict, out: Path):
    """Bar chart showing % performance retained under distractors."""
    methods = METHOD_ORDER
    retentions, errs = [], []

    for m in methods:
        clean_r = final_reward(data[('clean', m, 1)]) if ('clean', m, 1) in data else None
        dist_vals = [final_reward(data[('distractor', m, s)])
                     for s in [1, 2, 3] if ('distractor', m, s) in data]
        if clean_r and dist_vals:
            pcts = [v / clean_r * 100 for v in dist_vals]
            retentions.append(np.mean(pcts))
            errs.append(np.std(pcts) if len(pcts) > 1 else 0)
        else:
            retentions.append(0); errs.append(0)

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(methods))
    colors = [METHOD_COLORS[m] for m in methods]

    bars = ax.bar(x, retentions, color=colors, alpha=0.82,
                  yerr=errs, capsize=5, error_kw={'elinewidth': 1.5})

    ax.axhline(100, color='black', lw=1.2, ls='--', alpha=0.5, label='100% (no drop)')
    for bar, v, e in zip(bars, retentions, errs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + e + 0.8,
                f'{v:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[m] for m in methods], rotation=15, ha='right')
    ax.set_ylabel('Performance Retention (Distractor / Clean × 100%)')
    ax.set_title('Distractor Robustness — Cartpole Swingup', fontsize=13)
    ax.set_ylim(0, max(retentions) * 1.2)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    save(fig, out / 'fig4_robustness.png')


# ======================================================================
# Figure 5 & 6 — Alpha sensitivity for Mod A and Mod B
# ======================================================================

def fig_alpha_sensitivity(alpha_data: dict, mod: str, out: Path):
    """Learning curves for each alpha value of a given modification."""
    alphas = sorted(set(a for (m, a) in alpha_data if m == mod))
    if not alphas:
        print(f'  No alpha data for {mod}')
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    crashed = []

    for alpha in alphas:
        key = (mod, alpha)
        if key not in alpha_data:
            continue
        df = alpha_data[key]
        r_final = final_reward(df)
        color = ALPHA_COLORS.get(alpha, '#888')

        if r_final < 400:   # training collapsed
            crashed.append(alpha)
            # Still show the curve but dashed
            ax.plot(df['frame'], df['episode_reward'],
                    color=color, lw=1.5, ls='--', alpha=0.6,
                    label=f'α={alpha} (collapsed)')
        else:
            ax.plot(df['frame'], df['episode_reward'],
                    color=color, lw=2.0, label=f'α={alpha}')

    mod_label = 'Consistency (Mod A)' if mod == 'modA' else 'Contrastive (Mod B)'
    ax.set_title(f'α Sensitivity — {mod_label}', fontsize=13)
    ax.set_xlabel('Environment Frames')
    ax.set_ylabel('Episode Reward')
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_M))
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    if crashed:
        ax.annotate(f'α={crashed[0]} collapsed (training instability)',
                    xy=(0.03, 0.15), xycoords='axes fraction',
                    fontsize=8, color='gray', style='italic')
    fig.tight_layout()
    save(fig, out / f'fig5_alpha_{mod}.png')


# ======================================================================
# Figure 7 — Overlay: clean vs distractor per method
# ======================================================================

def fig_clean_vs_distractor(data: dict, out: Path):
    """4-panel plot: one per method, solid=clean, dashed=distractor."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for ax, method in zip(axes, METHOD_ORDER):
        color = METHOD_COLORS[method]

        # Clean (seed=1)
        if ('clean', method, 1) in data:
            df = data[('clean', method, 1)]
            ax.plot(df['frame'], df['episode_reward'],
                    color=color, lw=2.2, ls='-', label='Clean (seed=1)')

        # Distractor (mean ± std, 3 seeds)
        dfs = [data[('distractor', method, s)]
               for s in [1, 2, 3] if ('distractor', method, s) in data]
        if dfs:
            frames, mean, std = aggregate_seeds(dfs)
            if frames is not None:
                ax.plot(frames, mean, color=color, lw=2.2, ls='--',
                        label='Distractor (mean±std)')
                ax.fill_between(frames, mean - std, mean + std,
                                color=color, alpha=0.2)

        ax.set_title(METHOD_LABELS[method], fontsize=11)
        ax.set_xlabel('Environment Frames')
        ax.set_ylabel('Episode Reward')
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_M))
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Clean vs Distractor — All Methods (Cartpole Swingup)',
                 fontsize=14, y=1.01)
    fig.tight_layout()
    save(fig, out / 'fig6_clean_vs_distractor.png')


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='figures')
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print('Loading core experiments...')
    data = load_core_data()
    print(f'  Loaded {len(data)} runs')

    print('Loading alpha sweep experiments...')
    alpha_data = load_alpha_data()
    print(f'  Loaded {len(alpha_data)} alpha runs')

    print('\nGenerating figures...')
    fig_distractor_curves(data, out)
    fig_clean_curves(data, out)
    fig_final_performance(data, out)
    fig_robustness(data, out)
    fig_alpha_sensitivity(alpha_data, 'modA', out)
    fig_alpha_sensitivity(alpha_data, 'modB', out)
    fig_clean_vs_distractor(data, out)
    print('\nDone. All figures saved to', out)


if __name__ == '__main__':
    main()
