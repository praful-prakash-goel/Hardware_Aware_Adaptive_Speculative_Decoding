import matplotlib
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import os
import sys
from .benchmark_tps import RESULTS_DIR

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHIA_BENCHMARKS_PATH = os.path.join(RESULTS_DIR, "benchmarks_pythia.csv")
SMOL_BENCHMARKS_PATH = os.path.join(RESULTS_DIR, "benchmarks_smol.csv")
SMOL2_BENCHMARKS_PATH = os.path.join(RESULTS_DIR, "benchmarks_smol2.csv")
PYTHIA_STRESS_PATH = os.path.join(RESULTS_DIR, "stress_test_pythia.csv")
SMOL_STRESS_PATH = os.path.join(RESULTS_DIR, "stress_test_smol.csv")
SMOL2_STRESS_PATH = os.path.join(RESULTS_DIR, "stress_test_smol2.csv")
SMOL_PROFILE_PATH = os.path.join(RESULTS_DIR, "smollm_profile.csv")

PLOT_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# Standardized colours
edge_color= "#949494"
pythia_color = "#548235"
smol_color = "#9333EA"
smol2_color = "#E46565"

def get_ratio_and_peak(df, main_model, draft_model):
    main_tps = df[(df['method'] == f'Main {main_model}') & (df['cache'] == True)]['tps'].values[0]
    draft_tps = df[(df['method'] == f'Draft {draft_model}') & (df['cache'] == True)]['tps'].values[0]
    ratio = draft_tps / main_tps
    peak_sp = df[df['method'] == 'speculative']['speedup'].max()
    return ratio, peak_sp


def plot_graphs():
    # Load df for Pythia model
    if os.path.exists(PYTHIA_BENCHMARKS_PATH) and os.path.exists(PYTHIA_STRESS_PATH):
        df_pythia = pd.read_csv(PYTHIA_BENCHMARKS_PATH)
        df_pythia = df_pythia.drop(columns=['Unnamed: 0'])
        
        stress_df_pythia = pd.read_csv(PYTHIA_STRESS_PATH)
        stress_df_pythia = stress_df_pythia.drop(columns=['Unnamed: 0'])
        stress_df_pythia = stress_df_pythia.pivot(index='configuration', columns='context_length', values='tps')
    else:
        print(f">> No csv file found at {PYTHIA_BENCHMARKS_PATH} and {PYTHIA_STRESS_PATH}")
        sys.exit(1)
    
    # Load df for SmolLM model
    if os.path.exists(SMOL_BENCHMARKS_PATH) and os.path.exists(SMOL_STRESS_PATH):
        df_smol = pd.read_csv(SMOL_BENCHMARKS_PATH)
        df_smol = df_smol.drop(columns=['Unnamed: 0'])
        
        stress_df_smol = pd.read_csv(SMOL_STRESS_PATH)
        stress_df_smol = stress_df_smol.drop(columns=['Unnamed: 0'])
        stress_df_smol = stress_df_smol.pivot(index='configuration', columns='context_length', values='tps')

    else:
        print(f">> No csv file found at {SMOL_BENCHMARKS_PATH} and {SMOL_STRESS_PATH}")
        sys.exit(1)
    
    # Load df for SmolLM2 model
    if os.path.exists(SMOL2_BENCHMARKS_PATH) and os.path.exists(SMOL2_STRESS_PATH):
        df_smol2 = pd.read_csv(SMOL2_BENCHMARKS_PATH)
        df_smol2 = df_smol2.drop(columns=['Unnamed: 0'])
        
        stress_df_smol2 = pd.read_csv(SMOL2_STRESS_PATH)
        stress_df_smol2 = stress_df_smol2.drop(columns=['Unnamed: 0'])
        stress_df_smol2 = stress_df_smol2.pivot(index='configuration', columns='context_length', values='tps')

    else:
        print(f">> No csv file found at {SMOL2_BENCHMARKS_PATH} and {SMOL2_STRESS_PATH}")
        sys.exit(1)

    if os.path.exists(SMOL_PROFILE_PATH):
        df_smol_profile = pd.read_csv(SMOL_PROFILE_PATH)
        df_smol_profile = df_smol_profile.drop(columns=['Unnamed: 0'])

    else:
        print(f">> No csv file found at {SMOL_PROFILE_PATH}")
        sys.exit(1)
    
    # Get unique gamma values used in experiments
    gammas = df_smol['gamma'].dropna().unique()

    # df containing baseline models benchmark results
    df_pythia_baseline = df_pythia[df_pythia['method'].isin(['Main pythia-1B', 'Draft pythia-160M'])]
    df_smol_baseline = df_smol[df_smol['method'].isin(['Main smollm-1.7B', 'Draft smollm-135M'])]
    df_smol2_baseline = df_smol2[df_smol2['method'].isin(['Main smollm2-1.7B', 'Draft smollm2-135M'])]

    df_baseline = pd.concat([df_pythia_baseline, df_smol_baseline, df_smol2_baseline])
    df_baseline = df_baseline.pivot(index='cache', columns='method', values='tps') \
                .reindex(columns=['Main pythia-1B', 'Draft pythia-160M', 'Main smollm-1.7B', 'Draft smollm-135M', 'Main smollm2-1.7B', 'Draft smollm2-135M'])

    # Partition pythia df based on cache usage
    df_pythia_cache = df_pythia[(df_pythia['method'] == 'speculative') & (df_pythia['cache'] == True)]
    df_pythia_no_cache = df_pythia[(df_pythia['method'] == 'speculative') & (df_pythia['cache'] == False)]
    
    # Partition smol df based on cache usage
    df_smol_cache = df_smol[(df_smol['method'] == 'speculative') & (df_smol['cache'] == True)]
    df_smol_no_cache = df_smol[(df_smol['method'] == 'speculative') & (df_smol['cache'] == False)]
    
    # Partition smol2 df based on cache usage
    df_smol2_cache = df_smol2[(df_smol2['method'] == 'speculative') & (df_smol2['cache'] == True)]
    df_smol2_no_cache = df_smol2[(df_smol2['method'] == 'speculative') & (df_smol2['cache'] == False)]

    # Take only the cached rows from profile data of smollm
    df_smol_profile_cache = df_smol_profile[df_smol_profile['use_cache'] == True]

    # Graph 1: Baseline Tokens Per Second (TPS)
    # Assign draft colors
    pythia_draft = "#A9D18E"  # lighter green
    smol_draft = "#C990F5"  # lighter purple
    smol2_draft = "#F0AAAA"  # lighter red

    col_colors = [
        pythia_color, pythia_draft,
        smol_color, smol_draft,
        smol2_color, smol2_draft
    ]

    fig, ax = plt.subplots(figsize=(5, 3.5))

    df_baseline.plot(
        kind='bar',
        rot=0,
        ax=ax,
        color=col_colors,
        edgecolor='black',
        linewidth=0.5
    )

    ax.set_ylabel('Tokens Per Second', fontsize='13')
    ax.set_xlabel('cache', fontsize='15')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3), ncol=3, frameon=False, fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "baseline_tps.pdf"), bbox_inches='tight')

    # Graph2: Speedup vs gamma for all models (cache on and off)
    fig, ax = plt.subplots(figsize=(6.5, 3.8))

    # Cache ON — solid, full opacity
    ax.plot(gammas, df_pythia_cache['speedup'], color=pythia_color, marker='o', linewidth=2, markersize=7,
            label='Pythia (cache ON)')
    ax.plot(gammas, df_smol_cache['speedup'], color=smol_color, marker='s', linewidth=2, markersize=7,
            label='SmolLM (cache ON)')
    ax.plot(gammas, df_smol2_cache['speedup'], color=smol2_color, marker='D', linewidth=2, markersize=7,
            label='SmolLM2 (cache ON)')

    # Cache OFF — dashed, alpha=0.65
    ax.plot(gammas, df_pythia_no_cache['speedup'], color=pythia_color, marker='o', linewidth=1.5, markersize=6,
            linestyle='--', alpha=0.65, label='Pythia (cache OFF)')
    ax.plot(gammas, df_smol_no_cache['speedup'], color=smol_color, marker='s', linewidth=1.5, markersize=6,
            linestyle='--', alpha=0.65, label='SmolLM (cache OFF)')
    ax.plot(gammas, df_smol2_no_cache['speedup'], color=smol2_color, marker='D', linewidth=1.5, markersize=6,
            linestyle='--', alpha=0.65, label='SmolLM2 (cache OFF)')

    # Baseline
    ax.axhline(1.0, color='#444444', linestyle=':', linewidth=1.5, label='AR Baseline (1.0×)')

    # Annotate peaks for cache on
    for df, color, xoffset in [
        (df_pythia_cache, pythia_color, 0.2),
        (df_smol_cache, smol_color, 0.2),
        (df_smol2_cache, smol2_color, 0.2),
    ]:
        peak_idx = df['speedup'].to_numpy().argmax()
        peak_gamma = df['gamma'].iloc[peak_idx]
        peak_val = df['speedup'].iloc[peak_idx]
        ax.annotate(f'{peak_val:.2f}×',
                    xy=(peak_gamma, peak_val),
                    xytext=(peak_gamma + xoffset, peak_val + 0.08),
                    fontsize=8.5, color=color, fontweight='bold')

    ax.set_ylabel('Speedup over AR Baseline', fontsize=10)
    ax.set_xlabel('\u03B3', fontsize=15)
    ax.set_xticks(gammas)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Two-row legend: cache ON top row, cache OFF bottom row
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:7], labels[:7],
              loc='upper center', bbox_to_anchor=(0.5, 1.22),
              ncol=4, frameon=False, fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "speedup_gamma.pdf"), bbox_inches='tight')

    # Graph 3: Draft/Main TPS ratio and peak speedup for each model family
    fig, ax = plt.subplots(figsize=(6.5, 4.0))

    # Shaded regions — use fixed x boundary at 1.0
    ax.axvspan(-0.5, 1.0, alpha=0.40, color='red', zorder=0)
    ax.axvspan(1.0, 13.0, alpha=0.25, color='green', zorder=0)

    # Reference lines
    ax.axvline(1.0, color='#444444', linestyle='--', linewidth=1.5, alpha=0.7, zorder=1)
    ax.axhline(1.0, color='#444444', linestyle='--', linewidth=1.5, alpha=0.7, zorder=1)

    # Region labels — using axes fraction coordinates so they never overlap data
    ax.text(0.45, 3.3, 'Draft\nbottleneck\nzone',
            fontsize=6, color='#cc0000', alpha=0.95,
            style='italic', ha='center', va='top')
    ax.text(1.7, 3.3, 'Speculative decoding beneficial zone',
            fontsize=8, color='#1a7a1a', alpha=0.95,
            style='italic', ha='left', va='top')

    # AR baseline label — anchored to left side so it doesn't get cut off
    ax.text(12.5, 1.03, 'AR Baseline (1.0×)',
            fontsize=8, color='#444444', va='bottom', ha='right', style='italic')

    # Draft = Main speed label — above the vertical line, not below
    ax.text(1.05, 3.85, 'Draft = Main speed',
            fontsize=8, color='#444444', va='top', ha='left', style='italic')

    pythia_ratio, pythia_peak = get_ratio_and_peak(df_pythia, 'pythia-1B', 'pythia-160M')
    smol_ratio, smol_peak = get_ratio_and_peak(df_smol, 'smollm-1.7B', 'smollm-135M')
    smol2_ratio, smol2_peak = get_ratio_and_peak(df_smol2, 'smollm2-1.7B', 'smollm2-135M')

    # Points — defined with label positions carefully chosen
    points = [
        (pythia_ratio, pythia_peak, pythia_color,
         'Pythia\n(1B + 160M)', (0.25, 0.10)),  # right and slightly up
        (smol_ratio, smol_peak, smol_color,
         'SmolLM\n(1.7B + 135M)', (-0.3, 0.22)),  # left of point
        (smol2_ratio, smol2_peak, smol2_color,
         'SmolLM2\n(1.7B + 135M)', (0.32, -0.15)),  # right and below
    ]

    for ratio, peak, color, label, (dx, dy) in points:
        ax.scatter(ratio, peak,
                   color=color, s=50, zorder=5,
                   edgecolors='white', linewidths=1.2)  # white edge makes color pop
        ax.annotate(label,
                    xy=(ratio, peak),
                    xytext=(ratio + dx, peak + dy),
                    fontsize=9.5, color=color, fontweight='bold',
                    va='center', ha='left')

    ax.set_xlabel('Draft / Main TPS Ratio  (KV-Cache ON)', fontsize=12)
    ax.set_ylabel('Peak Speedup over AR Baseline', fontsize=12)
    ax.set_xlim(-0.1, 13.0)
    ax.set_ylim(0.5, 4.0)  # extend down so red zone has visual weight below SmolLM2
    ax.set_xticks([0, 2, 4, 6, 8, 10, 12])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='both', alpha=0.2)

    plt.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "ratio_vs_speedup.pdf"), bbox_inches='tight')

    # Graph 4: Subplot for stress test smollm and smollm2
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.5), sharey=True)

    x_ticks = stress_df_smol.columns

    axes[0].plot(stress_df_smol.columns, stress_df_smol.iloc[0],
                 color="#444444", linestyle=':', marker='s', markersize=5,
                 label='SmolLM AR - KV Cache')
    axes[0].plot(stress_df_smol.columns, stress_df_smol.iloc[1],
                 color=smol_color, linestyle='-', marker='s', markersize=5, linewidth=1.8,
                 label='SmolLM Speculative - KV Cache')
    axes[0].plot(stress_df_smol.columns, stress_df_smol.iloc[2],
                 color=smol_color, linestyle='dashed', marker='s', markersize=5, alpha=0.7,
                 label='SmolLM Speculative - No KV Cache')

    axes[0].set_xlabel("Max tokens generated", fontsize=12)
    axes[0].set_ylabel("Tokens Per Second", fontsize=12)
    axes[0].set_xticks(x_ticks)
    axes[0].set_xticklabels(['32', '64', '128', '256', '512'])
    axes[0].tick_params(axis='x', labelsize=8.5)
    axes[0].legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), frameon=False, fontsize=8.5)

    axes[1].plot(stress_df_smol2.columns, stress_df_smol2.iloc[0],
                 color="#444444", linestyle=':', marker='D', markersize=5,
                 label='SmolLM2 AR - KV Cache')
    axes[1].plot(stress_df_smol2.columns, stress_df_smol2.iloc[1],
                 color=smol2_color, linestyle='-', marker='D', markersize=5, linewidth=1.8,
                 label='SmolLM2 Speculative - KV Cache')
    axes[1].plot(stress_df_smol2.columns, stress_df_smol2.iloc[2],
                 color=smol2_color, linestyle='dashed', marker='D', markersize=5, alpha=0.7,
                 label='SmolLM2 Speculative - No KV Cache')

    axes[1].set_xlabel("Max tokens generated", fontsize=12)
    axes[1].set_xticks(x_ticks)
    axes[1].tick_params(axis='x', labelsize=8.5)
    axes[1].legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), frameon=False, fontsize=8.5)

    plt.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "stress_test.pdf"), bbox_inches='tight')

    # Graph 6: Profiler breakdown of Smollm speculative with cache
    draft_speculation_pct = df_smol_profile_cache['draft_speculation_pct'].values
    target_verification_pct = df_smol_profile_cache['target_verification_pct'].values

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.5), sharey=True)  # sharey=True syncs y axis

    bar_h = 0.55
    y = list(range(len(gammas)))

    # Left subplot: stacked bar
    axes[0].barh(y, draft_speculation_pct, height=bar_h, color='#2196F3',
                 label='Draft Speculation', edgecolor='white', linewidth=0.4)
    axes[0].barh(y, target_verification_pct, height=bar_h, left=draft_speculation_pct,
                 color='#FF9800', label='Target Verification', edgecolor='white', linewidth=0.4)

    for i, (d, v) in enumerate(zip(draft_speculation_pct, target_verification_pct)):
        axes[0].text(d/2, i, f'{d:.0f}%',
                     ha='center', va='center', fontsize=8.5, color='white', fontweight='bold')
        axes[0].text(d + v/2, i, f'{v:.0f}%',
                     ha='center', va='center', fontsize=8.5, color='white', fontweight='bold')

    axes[0].annotate('Draft ≈ Verify\nat γ=7',
                     xy=(50, 4), xytext=(58, 4.2),
                     fontsize=8, color='#333333',
                     arrowprops=dict(arrowstyle='->', color='#333333', lw=1.2))

    axes[0].set_yticks(y)
    axes[0].set_yticklabels([f'γ = {g}' for g in gammas], fontsize=10)
    axes[0].set_xlabel('Percentage of Total Step Time (%)', fontsize=10)
    axes[0].set_xlim(0, 100)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)

    # Right subplot: total timeline
    total_ms = df_smol_profile_cache['mean_total_ms'].values

    axes[1].plot(total_ms, y,
                 color='#333333', marker='D', linewidth=2, markersize=6,
                 linestyle='-', label='Mean Total Step Time', zorder=6)

    min_idx = total_ms.argmin()
    axes[1].scatter(total_ms[min_idx], min_idx,
                    color='red', s=50, zorder=7,
                    edgecolors='darkred', linewidths=1.2)
    axes[1].annotate(f'Min: {int(min(total_ms)):,}ms\n(γ=5, optimal)',
                     xy=(total_ms[min_idx], min_idx),
                     xytext=(total_ms[min_idx] + 1200, min_idx - 0.9),
                     fontsize=7.5, color='#333333',
                     arrowprops=dict(arrowstyle='->', color='#333333', lw=1.0))

    axes[1].set_yticks(y)
    axes[1].set_yticklabels([f'γ = {g}' for g in gammas], fontsize=10)
    axes[1].set_xlabel('Mean Total Step Time (ms)', fontsize=10)

    x_min = min(total_ms)
    x_max = max(total_ms)
    padding = (x_max - x_min) * 0.1 if x_max != x_min else x_min * 0.1

    axes[1].set_xlim(max(0, x_min - padding), x_max + padding)
    axes[1].xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f'{int(x / 1000)}k')
    )
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)

    # Shared legend at top
    handles1, labels1 = axes[0].get_legend_handles_labels()
    handles2, labels2 = axes[1].get_legend_handles_labels()
    fig.legend(handles1 + handles2, labels1 + labels2,
               loc='upper center', bbox_to_anchor=(0.5, 1.08),
               fontsize=8.5, frameon=True, framealpha=0.9,
               edgecolor='lightgray', ncol=3)

    # Footnote
    fig.text(0.5, -0.04,
             '* Cache Rollback, Comparison Logic, Tensor Updates each < 0.1% — not shown.',
             ha='center', fontsize=7.5, color='gray', style='italic')

    plt.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "smollm_profile.pdf"), bbox_inches='tight')

    print(f">> Plots saved at {PLOT_DIR}")
    
if __name__ == '__main__':
    plot_graphs()