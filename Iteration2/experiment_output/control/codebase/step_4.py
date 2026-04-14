# filename: codebase/step_4.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, chi2
import time
import os

def mahalanobis(x, mu, inv_cov):
    diff = x - mu
    return np.sqrt(np.dot(np.dot(diff, inv_cov), diff.T))

if __name__ == '__main__':
    data_dir = 'data/'
    train_path = os.path.join(data_dir, 'train_dataset_step2.csv')
    test_path = os.path.join(data_dir, 'test_dataset_step2.csv')
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    stable_train_df = train_df[train_df['is_stable'] == True].copy()
    corr, p_value = spearmanr(stable_train_df['M_soc_proxy'], stable_train_df['G_vrh'])
    print('--- Spearman Correlation (Stable Population) ---')
    print('Population size: ' + str(len(stable_train_df)))
    print('Feature: M_soc_proxy vs Target: G_vrh')
    print('Spearman R: ' + str(round(corr, 4)))
    print('p-value: ' + str(round(p_value, 4)))
    if p_value < 0.05:
        print('Result: Statistically significant correlation.')
    else:
        print('Result: Correlation is not statistically significant at alpha=0.05.')
    plt.rcParams['text.usetex'] = False
    plt.figure(figsize=(8, 6))
    plt.scatter(stable_train_df['M_soc_proxy'], stable_train_df['G_vrh'], alpha=0.7, edgecolors='k', s=50)
    plt.xlabel('M_soc_proxy (Standardized)', fontsize=12)
    plt.ylabel('Shear Modulus G_vrh (GPa)', fontsize=12)
    plt.title('M_soc_proxy vs G_vrh for Stable TMDs', fontsize=14)
    textstr = 'Spearman R = ' + str(round(corr, 3)) + '\np-value = ' + str(format(p_value, '.2e'))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=props)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    timestamp = int(time.time())
    scatter_filename = 'scatter_msoc_gvrh_1_' + str(timestamp) + '.png'
    scatter_filepath = os.path.join(data_dir, scatter_filename)
    plt.savefig(scatter_filepath, dpi=300)
    plt.close()
    print('\nScatter plot saved to ' + scatter_filepath)
    full_features = ['X_atomic_radius', 'crystal_system_encoded', 'M_group', 'd_band_filling', 'dos_at_fermi', 'M_soc_proxy', 'en_difference', 'magnetic_ordering_encoded', 'phase_encoded', 'imputation_uncertainty', 'is_dos_missing']
    X_stable = stable_train_df[full_features].values
    mu = np.mean(X_stable, axis=0)
    cov = np.cov(X_stable, rowvar=False)
    inv_cov = np.linalg.pinv(cov)
    rank = np.linalg.matrix_rank(cov)
    X_test = test_df[full_features].values
    distances = [mahalanobis(x, mu, inv_cov) for x in X_test]
    test_df['mahalanobis_distance'] = distances
    threshold_sq = chi2.ppf(0.95, df=rank)
    ood_threshold = np.sqrt(threshold_sq)
    test_df['is_ood'] = test_df['mahalanobis_distance'] > ood_threshold
    num_ood = test_df['is_ood'].sum()
    print('\n--- Mahalanobis Distance & OOD Analysis ---')
    print('Reference population: Stable training samples (N=' + str(len(stable_train_df)) + ')')
    print('Target population: Metastable candidates (N=' + str(len(test_df)) + ')')
    print('Covariance matrix rank: ' + str(rank) + ' (out of ' + str(len(full_features)) + ' features)')
    print('OOD Threshold (95% Chi-square): ' + str(round(ood_threshold, 4)))
    print('Number of OOD candidates: ' + str(num_ood) + ' (' + str(round(num_ood / len(test_df) * 100, 2)) + '%)')
    plt.figure(figsize=(8, 6))
    plt.hist(test_df['mahalanobis_distance'], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(ood_threshold, color='red', linestyle='dashed', linewidth=2, label='OOD Threshold (' + str(round(ood_threshold, 2)) + ')')
    plt.xlabel('Mahalanobis Distance', fontsize=12)
    plt.ylabel('Count of Metastable Candidates', fontsize=12)
    plt.title('Distribution of Mahalanobis Distances for Metastable TMDs', fontsize=14)
    plt.legend(loc='upper right', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    hist_filename = 'hist_mahalanobis_1_' + str(timestamp) + '.png'
    hist_filepath = os.path.join(data_dir, hist_filename)
    plt.savefig(hist_filepath, dpi=300)
    plt.close()
    print('Histogram saved to ' + hist_filepath)
    corr_stats = pd.DataFrame({'feature1': ['M_soc_proxy'], 'feature2': ['G_vrh'], 'spearman_r': [corr], 'p_value': [p_value], 'n_samples': [len(stable_train_df)]})
    corr_filepath = os.path.join(data_dir, 'correlation_statistics_step4.csv')
    corr_stats.to_csv(corr_filepath, index=False)
    print('\nCorrelation statistics saved to ' + corr_filepath)
    dist_metrics = test_df[['material_id', 'formula', 'mahalanobis_distance', 'is_ood']].copy()
    dist_filepath = os.path.join(data_dir, 'mahalanobis_distances_step4.csv')
    dist_metrics.to_csv(dist_filepath, index=False)
    print('Distance metrics saved to ' + dist_filepath)