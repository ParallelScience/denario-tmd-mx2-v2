# filename: codebase/step_4.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np
import json
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from step_2 import train_model, set_seed
from matplotlib.lines import Line2D

def main():
    set_seed(42)
    results_df = pd.read_csv('data/sensitivity_analysis_results.csv')
    orig_df = pd.read_csv('/home/node/work/projects/materials_project_v1/tmd_data_enriched.csv')
    stable_df = orig_df[orig_df['is_stable'] == True]
    stable_g_vrh = stable_df['G_vrh'].dropna()
    viability_threshold = stable_g_vrh.mean() + stable_g_vrh.std()
    atomic_masses = {'Ti': 47.867, 'Zr': 91.224, 'Hf': 178.49, 'V': 50.9415, 'Nb': 92.906, 'Ta': 180.947, 'Cr': 51.996, 'Mo': 95.95, 'W': 183.84, 'Mn': 54.938, 'Fe': 55.845, 'Co': 58.933, 'Ni': 58.693, 'S': 32.06, 'Se': 78.971, 'Te': 127.6}
    def get_density(row):
        m_mass = atomic_masses[row['metal']]
        x_mass = atomic_masses[row['chalcogen']]
        formula_mass = m_mass + 2 * x_mass
        density = formula_mass / (3 * 0.6022 * row['volume_per_atom'])
        return density
    results_df['density_g_cm3'] = results_df.apply(get_density, axis=1)
    results_df['specific_shear_modulus'] = results_df['G_vrh_pred_mean'] / results_df['density_g_cm3']
    unknown_candidates = results_df[results_df['is_known_elasticity'] == False].copy()
    unknown_candidates['meets_eah_criteria'] = unknown_candidates['energy_above_hull'] < 0.05
    unknown_candidates['meets_viability'] = unknown_candidates['G_vrh_pred_mean'] > viability_threshold
    unknown_candidates['viability_score'] = unknown_candidates['G_vrh_pred_mean']
    final_ranked_list = unknown_candidates.sort_values(by=['meets_eah_criteria', 'meets_viability', 'theoretical', 'viability_score'], ascending=[False, False, False, False])
    top_10 = final_ranked_list.head(10).copy()
    known_df = pd.read_csv('data/known_df_processed.csv')
    unknown_df = pd.read_csv('data/unknown_df_processed.csv')
    full_df = pd.concat([known_df, unknown_df], ignore_index=True)
    with open('data/feature_list.json', 'r') as f:
        feature_list = json.load(f)
    X = full_df[feature_list].values
    y_reg = full_df['G_vrh_norm'].values
    y_clf = full_df['is_stable'].values
    reg_mask = ~np.isnan(y_reg)
    y_reg_safe = np.nan_to_num(y_reg, nan=0.0)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_reg_tensor = torch.tensor(y_reg_safe, dtype=torch.float32)
    y_clf_tensor = torch.tensor(y_clf, dtype=torch.float32)
    reg_mask_tensor = torch.tensor(reg_mask, dtype=torch.float32)
    protected_features = ['d_band_filling', 'en_difference']
    protected_indices = [feature_list.index(f) for f in protected_features if f in feature_list]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    z_scaler = StandardScaler()
    orig_known = orig_df[orig_df['G_vrh'].notna()].copy()
    g_vrh_z = z_scaler.fit_transform(orig_known[['G_vrh']])
    minmax_scaler = MinMaxScaler()
    minmax_scaler.fit(g_vrh_z)
    def unnormalize_g_vrh(g_vrh_norm):
        shape = g_vrh_norm.shape
        g_vrh_z_un = minmax_scaler.inverse_transform(g_vrh_norm.reshape(-1, 1))
        g_vrh_un = z_scaler.inverse_transform(g_vrh_z_un)
        return g_vrh_un.reshape(shape)
    orig_full = pd.concat([orig_known, orig_df[orig_df['G_vrh'].isna()]], ignore_index=True)
    full_metals = orig_full['metal'].values
    unique_metals_in_top10 = top_10['metal'].unique()
    metal_models = {}
    for metal in unique_metals_in_top10:
        keep_mask = full_metals != metal
        X_train = X_tensor[keep_mask]
        y_reg_train = y_reg_tensor[keep_mask]
        y_clf_train = y_clf_tensor[keep_mask]
        reg_mask_train = reg_mask_tensor[keep_mask]
        model = train_model(X_train, y_reg_train, y_clf_train, reg_mask_train, protected_indices, epochs=300, lr=0.005, device=device)
        model.eval()
        metal_models[metal] = model
    top_10['G_vrh_pred_lomo'] = np.nan
    top_10['lomo_pct_change'] = np.nan
    top_10['robustness_flag'] = False
    for idx, row in top_10.iterrows():
        metal = row['metal']
        model = metal_models[metal]
        mat_idx = orig_full.index[orig_full['material_id'] == row['material_id']].tolist()[0]
        x_in = X_tensor[mat_idx:mat_idx+1].to(device)
        with torch.no_grad():
            reg_pred, _ = model(x_in)
            reg_pred_norm = reg_pred.cpu().numpy()
        pred_unnorm = unnormalize_g_vrh(reg_pred_norm)[0, 0]
        orig_pred = row['G_vrh_pred_mean']
        pct_change = abs(pred_unnorm - orig_pred) / orig_pred * 100
        top_10.at[idx, 'G_vrh_pred_lomo'] = pred_unnorm
        top_10.at[idx, 'lomo_pct_change'] = pct_change
        top_10.at[idx, 'robustness_flag'] = pct_change > 20.0
    final_ranked_list = final_ranked_list.merge(top_10[['material_id', 'G_vrh_pred_lomo', 'lomo_pct_change', 'robustness_flag']], on='material_id', how='left')
    final_list_path = os.path.join('data', 'prioritization_list.csv')
    final_ranked_list.to_csv(final_list_path, index=False)
    plt.figure(figsize=(10, 7))
    plt.rcParams['text.usetex'] = False
    sc = plt.scatter(results_df['energy_above_hull'], results_df['G_vrh_pred_mean'], c=results_df['G_vrh_pred_std'], cmap='viridis', alpha=0.8, edgecolors='w', s=60)
    high_corr = results_df[results_df['high_corr_sensitivity'] == True]
    if len(high_corr) > 0:
        plt.scatter(high_corr['energy_above_hull'], high_corr['G_vrh_pred_mean'], facecolors='none', edgecolors='red', s=100, linewidths=2, label='High-Corr Sensitivity')
    plt.axhline(y=viability_threshold, color='r', linestyle='--', label='Viability Threshold')
    plt.colorbar(sc, label='Prediction Uncertainty (Std Dev, GPa)')
    plt.xlabel('Energy Above Hull (eV/atom)')
    plt.ylabel('Predicted Shear Modulus G_vrh (GPa)')
    plt.title('Mechanical Viability Map of TMDs')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot1_path = os.path.join('data', 'mechanical_viability_map_4_' + timestamp + '.png')
    plt.savefig(plot1_path, dpi=300)
    plt.figure(figsize=(9, 6))
    plt.scatter(results_df['d_band_filling'], results_df['G_vrh_pred_mean'], c=results_df['is_stable'].astype(int), cmap='coolwarm', alpha=0.7, edgecolors='k', s=60)
    plt.axhline(y=viability_threshold, color='r', linestyle='--', label='Viability Threshold')
    plt.xlabel('d-band Filling (fractional)')
    plt.ylabel('Predicted Shear Modulus G_vrh (GPa)')
    plt.title('Shear Modulus vs. d-band Filling')
    custom_lines = [Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.coolwarm(0.0), markersize=10, markeredgecolor='k'), Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.coolwarm(1.0), markersize=10, markeredgecolor='k'), Line2D([0], [0], color='r', linestyle='--')]
    plt.legend(custom_lines, ['Metastable', 'Stable', 'Viability Threshold'])
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plot2_path = os.path.join('data', 'g_vrh_vs_dband_filling_4_' + timestamp + '.png')
    plt.savefig(plot2_path, dpi=300)

if __name__ == '__main__':
    main()