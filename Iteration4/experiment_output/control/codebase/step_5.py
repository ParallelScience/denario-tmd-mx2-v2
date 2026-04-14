# filename: codebase/step_5.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import joblib

mpl.rcParams['text.usetex'] = False

if __name__ == '__main__':
    data_dir = 'data/'
    full_df = pd.read_csv('/home/node/work/projects/materials_project_v1/tmd_data_enriched.csv')
    train_df = pd.read_csv(os.path.join(data_dir, 'training_dataset_step2.csv'))
    candidates_df = full_df[full_df['G_vrh'].isna()].copy()
    print('Number of metastable candidates (missing G_vrh): ' + str(len(candidates_df)))
    features_mah = ['d_band_filling', 'en_difference']
    X_train_mah = train_df[features_mah].values
    centroid = np.mean(X_train_mah, axis=0)
    cov_matrix = np.cov(X_train_mah, rowvar=False)
    inv_cov_matrix = np.linalg.inv(cov_matrix)
    def mahalanobis(x, mean, inv_cov):
        diff = x - mean
        return np.sqrt(np.dot(np.dot(diff, inv_cov), diff.T))
    distances = [mahalanobis(x, centroid, inv_cov_matrix) for x in candidates_df[features_mah].values]
    candidates_df['mahalanobis_dist'] = distances
    threshold = np.percentile(distances, 90)
    candidates_df['is_exotic'] = candidates_df['mahalanobis_dist'] > threshold
    num_exotic = candidates_df['is_exotic'].sum()
    print('\nOOD Detection (Mahalanobis Distance):')
    print('  - Features used: ' + ', '.join(features_mah))
    print('  - Centroid of training population: ' + str(np.round(centroid, 4)))
    print('  - 90th percentile distance threshold: ' + str(round(threshold, 4)))
    print('  - Number of flagged \'chemically exotic\' candidates: ' + str(num_exotic))
    with open(os.path.join(data_dir, 'selected_features.txt'), 'r') as f:
        selected_features = [line.strip() for line in f.readlines() if line.strip()]
    p5_dos = train_df['dos_at_fermi'].quantile(0.05)
    candidates_df['is_dos_missing'] = candidates_df['dos_at_fermi'].isna().astype(int)
    candidates_df['dos_at_fermi_imputed'] = candidates_df['dos_at_fermi'].fillna(p5_dos)
    for col in selected_features:
        if col.startswith('mag_'):
            mag_val = col.replace('mag_', '')
            candidates_df[col] = (candidates_df['magnetic_ordering'] == mag_val).astype(int)
        elif col not in candidates_df.columns:
            candidates_df[col] = train_df[col].median()
        elif candidates_df[col].isna().any():
            candidates_df[col] = candidates_df[col].fillna(train_df[col].median())
    X_cand = candidates_df[selected_features]
    model = joblib.load(os.path.join(data_dir, 'rf_model_step3.joblib'))
    probs = model.predict_proba(X_cand)[:, 1]
    candidates_df['predicted_viability_probability'] = probs
    output_csv = os.path.join(data_dir, 'metastable_candidates_predictions.csv')
    candidates_df.to_csv(output_csv, index=False)
    print('\nPredictions saved to ' + output_csv)
    plt.figure(figsize=(11, 8))
    sns.scatterplot(data=candidates_df, x='energy_above_hull', y='predicted_viability_probability', hue='is_exotic', style='phase', palette={True: 'red', False: 'blue'}, s=100, alpha=0.75)
    non_exotic = candidates_df[~candidates_df['is_exotic']].copy()
    pareto_front_indices = []
    for i, row1 in non_exotic.iterrows():
        dominated = False
        for j, row2 in non_exotic.iterrows():
            if i == j:
                continue
            better_energy = row2['energy_above_hull'] <= row1['energy_above_hull']
            better_prob = row2['predicted_viability_probability'] >= row1['predicted_viability_probability']
            strictly_better_energy = row2['energy_above_hull'] < row1['energy_above_hull']
            strictly_better_prob = row2['predicted_viability_probability'] > row1['predicted_viability_probability']
            if better_energy and better_prob and (strictly_better_energy or strictly_better_prob):
                dominated = True
                break
        if not dominated:
            pareto_front_indices.append(i)
    for idx in pareto_front_indices:
        row = candidates_df.loc[idx]
        plt.text(row['energy_above_hull'] + 0.005, row['predicted_viability_probability'], row['phase'], fontsize=10, color='black', fontweight='bold')
    plt.title('Pareto Front: Energy Above Hull vs. Predicted Viability Probability', fontsize=15)
    plt.xlabel('Energy Above Hull (eV/atom)', fontsize=13)
    plt.ylabel('Predicted Viability Probability', fontsize=13)
    plt.axhline(0.7, color='green', linestyle='--', alpha=0.6, label='Prob > 0.7')
    plt.axvline(0.05, color='purple', linestyle='--', alpha=0.6, label='Energy < 0.05')
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', title='Legend', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    timestamp = int(time.time())
    plot_filename = os.path.join(data_dir, 'pareto_front_5_' + str(timestamp) + '.png')
    plt.savefig(plot_filename, dpi=300)
    print('Pareto front plot saved to ' + plot_filename)