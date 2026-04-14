# filename: codebase/step_5.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

if __name__ == '__main__':
    plt.rcParams['text.usetex'] = False
    data_dir = 'data'
    uncertainty_path = os.path.join(data_dir, 'uncertainty_metrics.csv')
    processed_path = os.path.join(data_dir, 'processed_tmd_data.csv')
    original_path = '/home/node/work/projects/materials_project_v1/tmd_data_enriched.csv'
    df_unc = pd.read_csv(uncertainty_path)
    df_proc = pd.read_csv(processed_path)
    df_orig = pd.read_csv(original_path)
    df = df_unc.merge(df_proc[['material_id', 'pugh_ratio']], on='material_id', how='left')
    df = df.merge(df_orig[['material_id', 'phase']], on='material_id', how='left')
    df['viability_score'] = df['mean_pugh_prediction'] / df['epistemic_uncertainty']
    train_mask = df['pugh_ratio'].notna()
    dos_train_mean = df.loc[train_mask, 'dos_at_fermi'].mean()
    dos_train_std = df.loc[train_mask, 'dos_at_fermi'].std()
    dos_threshold = dos_train_mean + 3 * dos_train_std
    print('Training DOS at Fermi - Mean: ' + str(round(dos_train_mean, 4)) + ' states/eV, Std: ' + str(round(dos_train_std, 4)) + ' states/eV')
    print('Exclusion threshold (Mean + 3*Std): ' + str(round(dos_threshold, 4)) + ' states/eV')
    df['is_dos_outlier'] = df['dos_at_fermi'] > dos_threshold
    num_outliers = df['is_dos_outlier'].sum()
    print('Number of materials flagged as DOS outliers: ' + str(num_outliers))
    candidates_mask = df['pugh_ratio'].isna()
    df_candidates = df[candidates_mask].copy()
    print('Total candidates (no original elastic data): ' + str(len(df_candidates)))
    df_candidates_filtered = df_candidates[~df_candidates['is_dos_outlier']].copy()
    print('Candidates after excluding DOS outliers: ' + str(len(df_candidates_filtered)))
    df_candidates_ranked = df_candidates_filtered.sort_values(by='viability_score', ascending=False)
    output_cols = ['material_id', 'formula', 'metal', 'chalcogen', 'phase', 'mean_pugh_prediction', 'epistemic_uncertainty', 'viability_score', 'energy_above_hull', 'theoretical']
    ranked_path = os.path.join(data_dir, 'ranked_candidates.csv')
    df_candidates_ranked[output_cols].to_csv(ranked_path, index=False)
    print('Ranked candidates saved to: ' + ranked_path)
    print('\n--- Top 10 Candidates by Viability Score (Overall) ---')
    print(df_candidates_ranked[output_cols].head(10).to_string(index=False))
    print('\n--- Top 10 Candidates satisfying Stability-Robustness filter (energy_above_hull <= 0.05 eV/atom) and theoretical == True ---')
    robust_mask = (df_candidates_ranked['energy_above_hull'] <= 0.05) & (df_candidates_ranked['theoretical'] == True)
    print(df_candidates_ranked[robust_mask][output_cols].head(10).to_string(index=False))
    stable_train_mask = train_mask & (df['is_stable'] == True)
    median_pugh_stable = df.loc[stable_train_mask, 'pugh_ratio'].median()
    print('\nMedian Pugh\'s ratio of stable training materials: ' + str(round(median_pugh_stable, 4)))
    plt.figure(figsize=(10, 7))
    c_values = np.log10(np.clip(df['viability_score'], a_min=1e-5, a_max=None))
    vmin = c_values.min()
    vmax = c_values.max()
    plt.scatter(df[train_mask]['energy_above_hull'], df[train_mask]['mean_pugh_prediction'], c=c_values[train_mask], cmap='viridis', marker='s', alpha=0.8, edgecolor='k', s=50, vmin=vmin, vmax=vmax, label='Training Data')
    scatter = plt.scatter(df[~train_mask]['energy_above_hull'], df[~train_mask]['mean_pugh_prediction'], c=c_values[~train_mask], cmap='viridis', marker='o', alpha=0.8, edgecolor='k', s=50, vmin=vmin, vmax=vmax, label='Candidates')
    plt.axvline(x=0.05, color='red', linestyle='--', linewidth=2, label='Stability Threshold (0.05 eV/atom)')
    plt.axhline(y=median_pugh_stable, color='blue', linestyle=':', linewidth=2, label='Median Stable G/K (' + str(round(median_pugh_stable, 2)) + ')')
    cbar = plt.colorbar(scatter)
    cbar.set_label('Log10(Viability Score)', fontsize=12)
    plt.xlabel('Energy Above Hull (eV/atom)', fontsize=12)
    plt.ylabel("Predicted Pugh's Ratio (G/K)", fontsize=12)
    plt.title("2D Viability Map: Predicted Pugh's Ratio vs. Energy Above Hull", fontsize=14)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_path = os.path.join(data_dir, 'viability_map_1_' + timestamp + '.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print('\nViability map saved to ' + plot_path)
    print('\nSummary of Viability Scores for Filtered Candidates:')
    print(df_candidates_ranked['viability_score'].describe().to_string())
    top_20 = df_candidates_ranked.head(20)
    print('\nTop 20 Candidates - Theoretical vs Synthesized:')
    print(top_20['theoretical'].value_counts().to_string())