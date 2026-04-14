# filename: codebase/step_3.py
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
from step_2 import MultiTaskNet

def main():
    known_df = pd.read_csv('data/known_df_processed.csv')
    unknown_df = pd.read_csv('data/unknown_df_processed.csv')
    full_df = pd.concat([known_df, unknown_df], ignore_index=True)
    with open('data/feature_list.json', 'r') as f:
        feature_list = json.load(f)
    X = full_df[feature_list].values
    X_tensor = torch.tensor(X, dtype=torch.float32)
    z_scaler = StandardScaler()
    g_vrh_z = z_scaler.fit_transform(known_df[['G_vrh']])
    minmax_scaler = MinMaxScaler()
    minmax_scaler.fit(g_vrh_z)
    def unnormalize_g_vrh(g_vrh_norm):
        shape = g_vrh_norm.shape
        g_vrh_z_un = minmax_scaler.inverse_transform(g_vrh_norm.reshape(-1, 1))
        g_vrh_un = z_scaler.inverse_transform(g_vrh_z_un)
        return g_vrh_un.reshape(shape)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultiTaskNet(input_dim=len(feature_list)).to(device)
    model.load_state_dict(torch.load('data/final_multitask_model.pth', map_location=device))
    model.eval()
    print('Performing Monte Carlo sensitivity analysis (500 passes) with jittered inputs...')
    torch.manual_seed(42)
    np.random.seed(42)
    noise_std = 0.05
    mc_preds_norm = []
    X_tensor = X_tensor.to(device)
    with torch.no_grad():
        for _ in range(500):
            noise = torch.randn_like(X_tensor) * noise_std
            X_jittered = X_tensor + noise
            reg_pred, _ = model(X_jittered)
            mc_preds_norm.append(reg_pred.cpu().numpy())
    mc_preds_norm = np.array(mc_preds_norm)
    mc_preds_unnorm = unnormalize_g_vrh(mc_preds_norm)
    g_vrh_mean = mc_preds_unnorm.mean(axis=0).flatten()
    g_vrh_std = mc_preds_unnorm.std(axis=0).flatten()
    orig_df = pd.read_csv('data/tmd_data_enriched.csv')
    orig_known = orig_df[orig_df['G_vrh'].notna()].copy()
    orig_unknown = orig_df[orig_df['G_vrh'].isna()].copy()
    orig_full = pd.concat([orig_known, orig_unknown], ignore_index=True)
    assert (orig_full['material_id'].values == full_df['material_id'].values).all(), 'Row alignment mismatch!'
    d_band_filling = orig_full['d_band_filling'].values
    magnetic_ordering = orig_full['magnetic_ordering'].values
    high_corr_flag = (d_band_filling > 0.5) & np.isin(magnetic_ordering, ['FM', 'FiM'])
    results_df = pd.DataFrame({'material_id': orig_full['material_id'], 'formula': orig_full['formula'], 'metal': orig_full['metal'], 'chalcogen': orig_full['chalcogen'], 'volume_per_atom': orig_full['volume_per_atom'], 'G_vrh_pred_mean': g_vrh_mean, 'G_vrh_pred_std': g_vrh_std, 'high_corr_sensitivity': high_corr_flag, 'd_band_filling': d_band_filling, 'magnetic_ordering': magnetic_ordering, 'is_known_elasticity': orig_full['G_vrh'].notna(), 'energy_above_hull': orig_full['energy_above_hull'], 'is_stable': orig_full['is_stable'], 'theoretical': orig_full['theoretical']})
    results_path = os.path.join('data', 'sensitivity_analysis_results.csv')
    results_df.to_csv(results_path, index=False)
    print('Saved sensitivity analysis results to ' + results_path)
    print('\n=== Sensitivity Analysis Statistics ===')
    print('Mean G_vrh prediction std: ' + str(round(g_vrh_std.mean(), 4)) + ' GPa')
    print('Max G_vrh prediction std: ' + str(round(g_vrh_std.max(), 4)) + ' GPa')
    print('Number of materials with High-Correlation Sensitivity flag: ' + str(high_corr_flag.sum()))
    top_uncertain = results_df.sort_values('G_vrh_pred_std', ascending=False).head(5)
    print('\nTop 5 materials with highest prediction uncertainty:')
    print(top_uncertain[['formula', 'G_vrh_pred_mean', 'G_vrh_pred_std', 'high_corr_sensitivity']].to_string(index=False))
    plt.figure(figsize=(8, 6))
    plt.rcParams['text.usetex'] = False
    plt.scatter(d_band_filling[~high_corr_flag], g_vrh_std[~high_corr_flag], alpha=0.7, label='Normal', color='blue', edgecolors='k')
    plt.scatter(d_band_filling[high_corr_flag], g_vrh_std[high_corr_flag], alpha=0.9, label='High-Corr Sensitivity', color='red', marker='^', s=80, edgecolors='k')
    plt.xlabel('d-band Filling (fractional)')
    plt.ylabel('MC Prediction Uncertainty of G_vrh (GPa)')
    plt.title('Prediction Uncertainty vs. d-band Filling')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_filename = 'uncertainty_vs_dband_filling_3_' + timestamp + '.png'
    plot_path = os.path.join('data', plot_filename)
    plt.savefig(plot_path, dpi=300)
    print('\nPlot saved to ' + plot_path)

if __name__ == '__main__':
    main()