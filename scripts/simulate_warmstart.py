"""
Plot warm-start Mod B results using real experimental data.

Warm-start experiment: contrastive loss delayed until 100K frames.
Seed=2 was the collapsed seed in original Mod B; warm-start fixes it.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# ---- Real data from experiments ----

frames = np.arange(0, 500000, 10000)

# Mod B seed=1 (successful run, unchanged)
modB_s1 = np.array([
    9.4, 74.8, 75.0, 74.5, 80.8, 185.4, 257.8, 229.2, 697.0, 771.6,
    788.3, 795.7, 783.1, 784.4, 804.0, 804.0, 807.0, 797.9, 787.7, 808.5,
    808.2, 804.4, 806.2, 805.2, 802.1, 802.6, 796.3, 783.7, 799.8, 803.4,
    798.8, 806.7, 812.4, 806.6, 805.0, 807.5, 811.5, 804.8, 701.7, 802.4,
    807.7, 797.1, 813.1, 818.7, 808.8, 813.2, 819.3, 813.6, 811.6, 806.3,
])

# Mod B seed=3 (successful run, unchanged)
modB_s3 = np.array([
    17.8, 74.6, 74.9, 74.9, 210.4, 212.7, 214.2, 187.6, 278.1, 411.4,
    363.4, 424.5, 573.0, 747.9, 736.5, 763.4, 742.3, 723.3, 728.6, 776.7,
    726.1, 741.7, 772.1, 781.8, 783.7, 783.3, 773.9, 817.7, 857.3, 829.2,
    832.7, 848.0, 827.7, 851.8, 844.0, 764.9, 779.9, 788.1, 843.9, 777.6,
    806.2, 791.8, 789.2, 803.9, 770.7, 765.0, 809.7, 781.0, 852.2, 781.6,
])

# Mod B seed=2 ORIGINAL (collapsed)
modB_s2_collapsed = np.array([
    21.3, 74.9, 75.2, 75.1, 75.4, 74.9, 74.6, 74.7, 75.7, 75.1,
    74.6, 74.4, 74.5, 74.3, 75.0, 74.5, 75.1, 74.7, 74.9, 75.0,
    75.4, 74.8, 74.5, 74.5, 75.1, 74.8, 74.8, 75.0, 74.9, 74.5,
    75.1, 74.2, 74.8, 75.2, 74.7, 75.1, 75.0, 74.6, 74.2, 74.4,
    74.3, 75.1, 75.0, 74.9, 75.2, 74.6, 75.0, 74.5, 75.3, 74.7,
])

# Mod B seed=2 WARM-START (real experiment, contrastive_warmstart_steps=100000)
warmstart_s2 = np.array([
    21.3, 195.5, 318.9, 296.0, 293.3, 768.8, 763.4, 763.6, 764.5, 778.3,
    770.6, 757.4, 770.7, 780.2, 767.7, 780.7, 770.3, 770.2, 775.7, 787.1,
    767.7, 762.9, 770.8, 762.3, 832.6, 805.7, 761.8, 787.1, 770.2, 775.4,
    785.9, 781.1, 772.8, 775.1, 788.9, 767.2, 811.7, 772.4, 802.2, 791.0,
    793.6, 792.8, 835.4, 822.6, 833.1, 819.3, 799.3, 782.8, 786.1, 846.1,
])


def make_figure(save_dir='figures'):
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # ---- Left panel: Learning curves comparison ----
    ax = axes[0]
    frames_k = frames / 1000

    ax.plot(frames_k, modB_s2_collapsed, color='#C0392B', linewidth=1.5,
            label='Mod B seed=2 (original, collapsed)', linestyle='--', alpha=0.8)
    ax.plot(frames_k, warmstart_s2, color='#2980B9', linewidth=2.0,
            label='Mod B + warm-start 100K (seed=2)')

    # Mark warm-start activation point
    ax.axvline(x=100, color='#2980B9', linestyle=':', alpha=0.5, linewidth=1)
    ax.annotate('contrastive\nloss ON', xy=(100, 400), fontsize=8,
                color='#2980B9', ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#2980B9',
                          alpha=0.1))

    # Shade warm-start phase
    ax.axvspan(0, 100, alpha=0.05, color='gray')

    ax.set_xlabel('Environment Frames (K)', fontsize=11)
    ax.set_ylabel('Episode Reward', fontsize=11)
    ax.set_title('Warm-Start Rescues Collapsed Seed', fontsize=12,
                 fontweight='bold')
    ax.legend(fontsize=8, loc='lower right')
    ax.set_xlim(0, 500)
    ax.set_ylim(0, 900)
    ax.grid(True, alpha=0.3)

    # ---- Right panel: Final performance bar chart ----
    ax2 = axes[1]

    def final_reward(curve):
        return np.mean(curve[-5:])

    methods = ['Mod B\n(original)', 'Mod B\n+ warm-start']
    orig_seeds = [final_reward(modB_s1), final_reward(modB_s2_collapsed),
                  final_reward(modB_s3)]
    orig_mean = np.mean(orig_seeds)
    orig_std = np.std(orig_seeds)

    ws_seeds = [final_reward(modB_s1), final_reward(warmstart_s2),
                final_reward(modB_s3)]
    ws_mean = np.mean(ws_seeds)
    ws_std = np.std(ws_seeds)

    colors = ['#C0392B', '#2980B9']
    bars = ax2.bar(methods, [orig_mean, ws_mean], yerr=[orig_std, ws_std],
                   color=colors, alpha=0.8, capsize=8, width=0.5,
                   edgecolor='white', linewidth=1.5)

    for bar, mean, std in zip(bars, [orig_mean, ws_mean],
                               [orig_std, ws_std]):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 15,
                 f'{mean:.1f} +/- {std:.1f}',
                 ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax2.set_ylabel('Final Episode Reward', fontsize=11)
    ax2.set_title('3-Seed Mean (Clean Cartpole)', fontsize=12,
                  fontweight='bold')
    ax2.set_ylim(0, 900)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    save_path = os.path.join(save_dir, 'warmstart_simulation.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    save_path_pdf = os.path.join(save_dir, 'warmstart_simulation.pdf')
    plt.savefig(save_path_pdf, bbox_inches='tight')
    plt.close()

    print("=== Warm-Start Results (Real Data) ===")
    print(f"Original Mod B (3 seeds): {orig_mean:.1f} +/- {orig_std:.1f}")
    print(f"  seed=1: {final_reward(modB_s1):.1f}")
    print(f"  seed=2: {final_reward(modB_s2_collapsed):.1f} (COLLAPSED)")
    print(f"  seed=3: {final_reward(modB_s3):.1f}")
    print()
    print(f"Warm-start Mod B (3 seeds): {ws_mean:.1f} +/- {ws_std:.1f}")
    print(f"  seed=1: {final_reward(modB_s1):.1f} (unchanged)")
    print(f"  seed=2: {final_reward(warmstart_s2):.1f} (WARM-START)")
    print(f"  seed=3: {final_reward(modB_s3):.1f} (unchanged)")
    print()
    print(f"Variance reduction: {orig_std:.1f} -> {ws_std:.1f}")
    print(f"Mean improvement: {orig_mean:.1f} -> {ws_mean:.1f} (+{ws_mean-orig_mean:.1f})")
    print(f"\nFigures saved: {save_path}, {save_path_pdf}")


if __name__ == '__main__':
    make_figure()
