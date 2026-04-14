# filename: codebase/step_6.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import json
import time
import shap

def main():
    plt.rcParams['text.usetex'] = False
    data_dir = 'data'
    processed_data_path = os.path.join(data_dir, 'processed_tmd_data.csv')
    targets_path = os.path.join(data_dir, 'target_labels.csv')
    features_path = os.path.join(data_dir, 'training_features.json')
    model_path = os.path.join(data_dir, 'rf_model.joblib')
    df_processed = pd.read_csv(processed_data_path)
    df_targets = pd.read_csv(targets_path)
    with open(features_path, 'r') as f:
        features = json.load(f)
    rf_model = joblib.load(model_path)
    df = pd.merge(df_targets[['material_id', 'is_robust']], df_processed, on='material_id', how='inner')
    X = df[features]
    y = df['is_robust']
    print('Calculating SHAP values for training set...')
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X)
    shap_interaction = explainer.shap_interaction_values(X)
    if isinstance(shap_values, list):
        shap_values_class1 = shap_values[1]
    elif len(shap_values.shape) == 3:
        shap_values_class1 = shap_values[:, :, 1]
    else:
        shap_values_class1 = shap_values
    if isinstance(shap_interaction, list):
        shap_interaction_class1 = shap_interaction[1]
    elif len(shap_interaction.shape) == 4:
        shap_interaction_class1 = shap_interaction[:, :, :, 1]
    else:
        shap_interaction_class1 = shap_interaction
    timestamp = int(time.time())
    shap.summary_plot(shap_values_class1, X, show=False, plot_size=(10, 8))
    plt.title('SHAP Summary Plot (Feature Importance)')
    plt.tight_layout()
    summary_plot_path = os.path.join(data_dir, 'shap_summary_plot_6_' + str(timestamp) + '.png')
    plt.savefig(summary_plot_path, dpi=300)
    plt.close()
    print('SHAP summary plot saved to ' + summary_plot_path)
    idx_d = features.index('d_band_filling')
    idx_dos = features.index('dos_at_fermi')
    ni_all_df = df_processed[df_processed['metal'] == 'Ni'].copy()
    X_ni_all = ni_all_df[features]
    print('Calculating SHAP interaction values for Ni-based TMDs...')
    shap_interaction_all = explainer.shap_interaction_values(X_ni_all)
    if isinstance(shap_interaction_all, list):
        shap_interaction_all_class1 = shap_interaction_all[1]
    elif len(shap_interaction_all.shape) == 4:
        shap_interaction_all_class1 = shap_interaction_all[:, :, :, 1]
    else:
        shap_interaction_all_class1 = shap_interaction_all
    interaction_vals_all = shap_interaction_all_class1[:, idx_d, idx_dos] * 2
    ni_all_df['interaction_val'] = interaction_vals_all
    ni_all_df = pd.merge(ni_all_df, df_targets[['material_id', 'is_robust']], on='material_id', how='left')
    plt.figure(figsize=(8, 6))
    known_mask = ni_all_df['is_robust'].notna()
    if known_mask.sum() > 0:
        scatter = plt.scatter(ni_all_df.loc[known_mask, 'dos_at_fermi'], ni_all_df.loc[known_mask, 'interaction_val'], c=ni_all_df.loc[known_mask, 'is_robust'], cmap='coolwarm', s=150, edgecolor='k', vmin=0, vmax=1, label='Known is_robust')
        cbar = plt.colorbar(scatter)
        cbar.set_label('is_robust')
    unknown_mask = ni_all_df['is_robust'].isna()
    if unknown_mask.sum() > 0:
        plt.scatter(ni_all_df.loc[unknown_mask, 'dos_at_fermi'], ni_all_df.loc[unknown_mask, 'interaction_val'], c='lightgray', s=100, edgecolor='k', marker='s', label='Unknown (Metastable)')
    plt.axhline(0, color='gray', linestyle='--', alpha=0.7)
    plt.title('SHAP Interaction: d_band_filling & dos_at_fermi (All Ni-based TMDs)')
    plt.xlabel('dos_at_fermi (Standardized)')
    plt.ylabel('SHAP Interaction Value')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    interaction_plot_path = os.path.join(data_dir, 'shap_interaction_ni_6_' + str(timestamp) + '.png')
    plt.savefig(interaction_plot_path, dpi=300)
    plt.close()
    print('SHAP interaction plot saved to ' + interaction_plot_path)
    ni_all_df['pred_proba'] = rf_model.predict_proba(X_ni_all)[:, 1]
    print('\nNi-based TMDs Analysis (All Ni-based samples in dataset):')
    print('Material ID     | Pred Proba | d_band_filling (std) | dos_at_fermi (std) | is_robust')
    print('-' * 85)
    for _, row in ni_all_df.iterrows():
        m_id = str(row['material_id']).ljust(15)
        proba = str(round(row['pred_proba'], 4)).ljust(10)
        d_band = str(round(row['d_band_filling'], 4)).ljust(20)
        dos = str(round(row['dos_at_fermi'], 4)).ljust(18)
        is_rob = str(int(row['is_robust'])) if pd.notna(row['is_robust']) else 'Unknown'
        print(m_id + ' | ' + proba + ' | ' + d_band + ' | ' + dos + ' | ' + is_rob)
    print('-' * 85 + '\n')
    shap_values_path = os.path.join(data_dir, 'shap_values.npy')
    shap_interaction_path = os.path.join(data_dir, 'shap_interaction_values.npy')
    np.save(shap_values_path, shap_values_class1)
    np.save(shap_interaction_path, shap_interaction_class1)
    print('SHAP values saved to ' + shap_values_path)
    print('SHAP interaction values saved to ' + shap_interaction_path)

if __name__ == '__main__':
    main()