# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import os
import time

mpl.rcParams['text.usetex'] = False

def calculate_separation(v_stable, v_meta):
    mean_stable = v_stable.mean()
    var_stable = v_stable.var(ddof=1) if len(v_stable) > 1 else 0
    n_stable = len(v_stable)
    mean_meta = v_meta.mean()
    var_meta = v_meta.var(ddof=1) if len(v_meta) > 1 else 0
    n_meta = len(v_meta)
    denom = np.sqrt(var_stable/n_stable + var_meta/n_meta)
    if denom == 0:
        return 0
    return (mean_stable - mean_meta) / denom

if __name__ == '__main__':
    data_path = "/home/node/work/projects/materials_project_v1/tmd_data_enriched.csv"
    df = pd.read_csv(data_path)
    df_elastic = df.dropna(subset=['G_vrh']).copy()
    print("Loaded dataset and filtered for materials with elastic data. Count: " + str(len(df_elastic)))
    df_elastic['G_vrh_percentile'] = df_elastic['G_vrh'].rank(pct=True)
    alphas = np.linspace(0, 50, 500)
    t_stats = []
    stable_mask = df_elastic['is_stable'] == True
    meta_mask = df_elastic['is_stable'] == False
    for alpha in alphas:
        v_score = df_elastic['G_vrh_percentile'] * np.exp(-alpha * df_elastic['energy_above_hull'])
        v_stable = v_score[stable_mask]
        v_meta = v_score[meta_mask]
        stat = calculate_separation(v_stable, v_meta)
        t_stats.append(stat)
    best_alpha = alphas[np.argmax(t_stats)]
    max_stat = np.max(t_stats)
    print("\nCalibration of decay constant alpha:")
    print("  - Scanned range: 0 to 50")
    print("  - Optimal alpha found: " + str(round(best_alpha, 2)))
    print("  - Maximum t-statistic (separation): " + str(round(max_stat, 4)))
    print("Justification: The chosen alpha maximizes the Welch's t-test statistic between the V_score distributions of stable and metastable materials within the scanned range [0, 50]. This ensures thermodynamic stability significantly penalizes the score while maintaining numerical stability.")
    df_elastic['V_score'] = df_elastic['G_vrh_percentile'] * np.exp(-best_alpha * df_elastic['energy_above_hull'])
    median_v_stable = df_elastic.loc[stable_mask, 'V_score'].median()
    df_elastic['is_viable'] = (df_elastic['V_score'] > median_v_stable) & (df_elastic['energy_above_hull'] <= 0.1)
    print("\nTarget Definition:")
    print("  - Median V_score of stable population: " + str(round(median_v_stable, 4)))
    print("  - Number of viable materials: " + str(df_elastic['is_viable'].sum()) + " out of " + str(len(df_elastic)))
    print("  - Number of viable stable materials: " + str(df_elastic[stable_mask]['is_viable'].sum()))
    print("  - Number of viable metastable materials: " + str(df_elastic[meta_mask]['is_viable'].sum()))
    output_csv = "data/processed_tmd_data.csv"
    df_elastic.to_csv(output_csv, index=False)
    print("\nProcessed dataframe saved to " + output_csv)
    plt.figure(figsize=(10, 6))
    ax = sns.histplot(data=df_elastic, x='V_score', hue='is_stable', bins=20, kde=True, palette={True: 'blue', False: 'red'}, element='step', stat='density', common_norm=False)
    plt.title('Distribution of Mechanical Viability Score (V_score) by Stability', fontsize=14)
    plt.xlabel('Mechanical Viability Score (V_score) (dimensionless)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.axvline(median_v_stable, color='black', linestyle='--')
    y_min, y_max = ax.get_ylim()
    plt.text(median_v_stable + 0.02, y_max * 0.9, 'Stable Median\n(' + str(round(median_v_stable, 2)) + ')', color='black')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    timestamp = int(time.time())
    plot_filename = "data/vscore_distribution_1_" + str(timestamp) + ".png"
    plt.savefig(plot_filename, dpi=300)
    print("Plot saved to " + plot_filename)