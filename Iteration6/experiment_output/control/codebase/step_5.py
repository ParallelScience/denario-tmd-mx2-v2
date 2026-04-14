# filename: codebase/step_5.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import joblib

plt.rcParams['text.usetex'] = False

def main():
    data_dir = 'data'
    df_path = os.path.join(data_dir, 'processed_tmd_data.csv')
    unc_path = os.path.join(data_dir, 'uncertainty_metrics.csv')
    df = pd.read_csv(df_path)
    df_unc = pd.read_csv(unc_path)
    df = df.merge(df_unc, on='material_id')
    ensemble_path = os.path.join(data_dir, 'ensemble_models.joblib')
    ensemble_data = joblib.load(ensemble_path)
    dos_mean = ensemble_data['dos_mean']
    df['dos_at_fermi'] = df['dos_at_fermi'].fillna(dos_mean)
    dos_std = df['dos_at_fermi'].std()
    dos_threshold = dos_mean + 3 * dos_std
    initial_count = len(df)
    df = df[df['dos_at_fermi'] <= dos_threshold].copy()
    filtered_count = len(df)
    print('Filtered out ' + str(initial_count - filtered_count) + ' materials with dos_at_fermi > 3 std from mean (' + str(round(dos_threshold, 4)) + ').')
    print('Total materials remaining: ' + str(filtered_count))
    epsilon = 1e-8
    df['viability_score'] = df['mean_pred_pugh_ratio'] / (df['std_pred_pugh_ratio'] + epsilon)
    timestamp = str(int(datetime.now().timestamp()))
    plt.figure(figsize=(10, 8))
    mask_theo = (df['theoretical'] == True) | (df['theoretical'] == 1)
    mask_exp = (df['theoretical'] == False) | (df['theoretical'] == 0)
    plt.scatter(df.loc[mask_theo, 'energy_above_hull'], df.loc[mask_theo, 'mean_pred_pugh_ratio'], c='blue', label='Theoretical (Unobserved)', alpha=0.7, edgecolors='k', s=60)
    plt.scatter(df.loc[mask_exp, 'energy_above_hull'], df.loc[mask_exp, 'mean_pred_pugh_ratio'], c='orange', label='Experimental (Observed)', alpha=0.7, edgecolors='k', s=60)
    plt.axvline(x=0.05, color='red', linestyle='--', linewidth=2, label='Stability Threshold (0.05 eV/atom)')
    stable_mask = ((df['is_stable'] == True) | (df['is_stable'] == 1)) & df['pugh_ratio'].notna()
    stable_pugh_median = df.loc[stable_mask, 'pugh_ratio'].median()
    if pd.notna(stable_pugh_median):
        plt.axhline(y=stable_pugh_median, color='green', linestyle=':', linewidth=2, label='Stable Median G/K (' + str(round(stable_pugh_median, 2)) + ')')
    plt.xlabel('Energy Above Hull (eV/atom)', fontsize=12)
    plt.ylabel('Predicted Pugh\'s Ratio (G/K)', fontsize=12)
    plt.title('Viability Map: Predicted Pugh\'s Ratio vs. Thermodynamic Stability', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_filename = 'viability_map_1_' + timestamp + '.png'
    plot_path = os.path.join(data_dir, plot_filename)
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print('Viability map saved to ' + plot_path)
    candidates = df[df['pugh_ratio'].isna()].copy()
    candidates = candidates.sort_values(by='viability_score', ascending=False)
    ranked_path = os.path.join(data_dir, 'ranked_candidates.csv')
    candidates.to_csv(ranked_path, index=False)
    print('Ranked candidates saved to ' + ranked_path)
    top_10 = candidates.head(10)
    print('\n--- Top 10 Ranked Metastable Candidates (Overall) ---')
    print('Rank  | Formula    | Phase           | E_hull (eV)  | Pred G/K   | Uncertainty  | Score S    | Theoretical')
    print('-' * 105)
    for i, (_, row) in enumerate(top_10.iterrows(), 1):
        phase_cols = [c for c in df.columns if c.startswith('phase_') and row[c] == 1]
        phase_str = phase_cols[0].replace('phase_', '') if phase_cols else 'Unknown'
        print(str(i).ljust(5) + ' | ' + str(row['formula']).ljust(10) + ' | ' + phase_str.ljust(15) + ' | ' + str(round(row['energy_above_hull'], 4)).ljust(12) + ' | ' + str(round(row['mean_pred_pugh_ratio'], 4)).ljust(10) + ' | ' + str(round(row['std_pred_pugh_ratio'], 4)).ljust(12) + ' | ' + str(round(row['viability_score'], 4)).ljust(10) + ' | ' + str(row['theoretical']))
    prioritized = candidates[(candidates['energy_above_hull'] <= 0.05) & ((candidates['theoretical'] == True) | (candidates['theoretical'] == 1))]
    print('\n--- Top Prioritized Candidates (E_hull <= 0.05 & Theoretical) ---')
    if len(prioritized) > 0:
        print('Rank  | Formula    | Phase           | E_hull (eV)  | Pred G/K   | Uncertainty  | Score S')
        print('-' * 90)
        for i, (_, row) in enumerate(prioritized.head(10).iterrows(), 1):
            phase_cols = [c for c in df.columns if c.startswith('phase_') and row[c] == 1]
            phase_str = phase_cols[0].replace('phase_', '') if phase_cols else 'Unknown'
            print(str(i).ljust(5) + ' | ' + str(row['formula']).ljust(10) + ' | ' + phase_str.ljust(15) + ' | ' + str(round(row['energy_above_hull'], 4)).ljust(12) + ' | ' + str(round(row['mean_pred_pugh_ratio'], 4)).ljust(10) + ' | ' + str(round(row['std_pred_pugh_ratio'], 4)).ljust(12) + ' | ' + str(round(row['viability_score'], 4)).ljust(10))
    else:
        print('No candidates meet the strict prioritization criteria.')
    print('\n--- Leave-One-Metal-Out Sensitivity Check for Top 10 Candidates ---')
    print('Formula    | Metal  | Ensemble Pred   | LOGO CV Pred    | Absolute Diff')
    print('-' * 68)
    for _, row in top_10.iterrows():
        diff = abs(row['mean_pred_pugh_ratio'] - row['cv_pred_pugh_ratio'])
        print(str(row['formula']).ljust(10) + ' | ' + str(row['metal']).ljust(6) + ' | ' + str(round(row['mean_pred_pugh_ratio'], 4)).ljust(15) + ' | ' + str(round(row['cv_pred_pugh_ratio'], 4)).ljust(15) + ' | ' + str(round(diff, 4)).ljust(15))
    mean_diff = np.mean([abs(row['mean_pred_pugh_ratio'] - row['cv_pred_pugh_ratio']) for _, row in top_10.iterrows()])
    print('-' * 68)
    print('Mean Absolute Difference for Top 10: ' + str(round(mean_diff, 4)))

if __name__ == '__main__':
    main()