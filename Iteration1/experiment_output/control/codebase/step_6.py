# filename: codebase/step_6.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import time
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.inspection import PartialDependenceDisplay

plt.rcParams['text.usetex'] = False

def main():
    data_dir = 'data'
    timestamp = int(time.time())
    feat_imp_df = pd.read_csv(os.path.join(data_dir, 'feature_importance.csv'))
    all_cand_df = pd.read_csv(os.path.join(data_dir, 'all_candidates_predictions.csv'))
    prio_cand_df = pd.read_csv(os.path.join(data_dir, 'prioritized_candidates.csv'))
    elastic_df = pd.read_csv(os.path.join(data_dir, 'elastic_subset.csv'))
    print('--- Plot 1: Feature Importance ---')
    print('Top 3 features:')
    for i, row in feat_imp_df.head(3).iterrows():
        print('  ' + row['feature'] + ': ' + str(round(row['importance_mean'], 4)) + ' +/- ' + str(round(row['importance_std'], 4)))
    print('----------------------------------\n')
    plt.figure(figsize=(10, 8))
    feat_imp_df_sorted = feat_imp_df.sort_values('importance_mean', ascending=True)
    plt.barh(feat_imp_df_sorted['feature'], feat_imp_df_sorted['importance_mean'], xerr=feat_imp_df_sorted['importance_std'], capsize=4, color='skyblue', edgecolor='black')
    plt.xlabel('Mean AUPRC Decrease')
    plt.title('Permutation Feature Importance')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plot1_path = os.path.join(data_dir, 'feature_importance_1_' + str(timestamp) + '.png')
    plt.savefig(plot1_path, dpi=300)
    plt.close()
    print('Feature importance plot saved to ' + plot1_path)
    with open(os.path.join(data_dir, 'lomo_cv_results.json'), 'r') as f:
        lomo_results = json.load(f)
    hp_counts = {}
    for metal, res in lomo_results.items():
        hp = res['best_hp']
        hp_str = 'n_est=' + str(hp['n_estimators']) + ', max_depth=' + str(hp['max_depth']) + ', min_samples_split=' + str(hp['min_samples_split'])
        if hp_str not in hp_counts:
            hp_counts[hp_str] = {'count': 0, 'hp': hp, 'inner_auprc_sum': 0}
        hp_counts[hp_str]['count'] += 1
        hp_counts[hp_str]['inner_auprc_sum'] += res['inner_auprc']
    best_hp_str = max(hp_counts.keys(), key=lambda k: (hp_counts[k]['count'], hp_counts[k]['inner_auprc_sum']))
    best_hp = hp_counts[best_hp_str]['hp']
    knn_features = ['band_gap', 'd_band_filling', 'en_difference', 'M_soc_proxy']
    valid_train = elastic_df[~elastic_df['dos_at_fermi_imputed_flag']]
    if len(valid_train) > 0:
        knn = KNeighborsRegressor(n_neighbors=min(5, len(valid_train)))
        knn.fit(valid_train[knn_features], valid_train['dos_at_fermi'])
        impute_train_mask = elastic_df['dos_at_fermi_imputed_flag']
        if impute_train_mask.sum() > 0:
            elastic_df.loc[impute_train_mask, 'dos_at_fermi'] = knn.predict(elastic_df.loc[impute_train_mask, knn_features])
    else:
        elastic_df.loc[elastic_df['dos_at_fermi_imputed_flag'], 'dos_at_fermi'] = 0.0
    ohe_features = [c for c in elastic_df.columns if c.startswith('magnetic_ordering_') or c.startswith('phase_')]
    rf_features = ['d_band_filling', 'M_soc_proxy', 'en_difference', 'M_atomic_radius', 'X_atomic_radius', 'dos_at_fermi', 'crystal_system_ord'] + ohe_features
    rf_final = RandomForestClassifier(n_estimators=best_hp['n_estimators'], max_depth=best_hp['max_depth'], min_samples_split=best_hp['min_samples_split'], class_weight='balanced', random_state=42, n_jobs=1)
    rf_final.fit(elastic_df[rf_features], elastic_df['is_robust'])
    top3_features = feat_imp_df['feature'].head(3).tolist()
    fig, ax = plt.subplots(figsize=(15, 5))
    PartialDependenceDisplay.from_estimator(rf_final, elastic_df[rf_features], top3_features, ax=ax, grid_resolution=50)
    plt.suptitle('Partial Dependence Plots for Top 3 Features')
    plt.tight_layout()
    plot2_path = os.path.join(data_dir, 'pdp_top3_2_' + str(timestamp) + '.png')
    plt.savefig(plot2_path, dpi=300)
    plt.close()
    print('PDP plot saved to ' + plot2_path)
    chalcogens = all_cand_df['chalcogen'].unique()
    print('\n--- Plot 3: Scatter Plot Statistics ---')
    print('Total metastable candidates plotted: ' + str(len(all_cand_df)))
    for chalc in chalcogens:
        subset = all_cand_df[all_cand_df['chalcogen'] == chalc]
        print('  Chalcogen ' + chalc + ': ' + str(len(subset)) + ' candidates')
    print('---------------------------------------\n')
    plt.figure(figsize=(8, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for i, chalc in enumerate(chalcogens):
        subset = all_cand_df[all_cand_df['chalcogen'] == chalc]
        plt.scatter(subset['d_band_filling'], subset['prob_robust'], label=chalc, color=colors[i % len(colors)], alpha=0.7, edgecolors='k')
    plt.xlabel('d-band filling (standardized)')
    plt.ylabel('Predicted Probability of Robustness')
    plt.title('Robustness Probability vs. d-band filling for Metastable Candidates')
    plt.legend(title='Chalcogen')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plot3_path = os.path.join(data_dir, 'scatter_prob_vs_dband_3_' + str(timestamp) + '.png')
    plt.savefig(plot3_path, dpi=300)
    plt.close()
    print('Scatter plot saved to ' + plot3_path)
    high_priority = prio_cand_df[prio_cand_df['prob_robust'] > 0.75]
    stable_gvrh = elastic_df[elastic_df['is_stable'] == True]['G_vrh'].dropna()
    cand_gvrh = high_priority['pred_G_vrh'].dropna()
    print('\n--- Plot 4: Histogram Statistics ---')
    print('Stable Population (Actual G_vrh):')
    print('  Count: ' + str(len(stable_gvrh)))
    print('  Mean:  ' + str(round(stable_gvrh.mean(), 4)) + ' GPa')
    print('  Std:   ' + str(round(stable_gvrh.std(), 4)) + ' GPa')
    print('High-Priority Candidates (Predicted G_vrh):')
    print('  Count: ' + str(len(cand_gvrh)))
    print('  Mean:  ' + str(round(cand_gvrh.mean(), 4)) + ' GPa')
    print('  Std:   ' + str(round(cand_gvrh.std(), 4)) + ' GPa')
    print('------------------------------------\n')
    plt.figure(figsize=(8, 6))
    plt.hist(stable_gvrh, bins=15, alpha=0.6, label='Stable Population (Actual G_vrh)', density=True, color='blue', edgecolor='black')
    plt.hist(cand_gvrh, bins=15, alpha=0.6, label='High-Priority Candidates (Predicted G_vrh)', density=True, color='orange', edgecolor='black')
    plt.xlabel('Shear Modulus G_vrh (GPa)')
    plt.ylabel('Density')
    plt.title('G_vrh Distribution: Stable vs. High-Priority Candidates')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plot4_path = os.path.join(data_dir, 'histogram_gvrh_comparison_4_' + str(timestamp) + '.png')
    plt.savefig(plot4_path, dpi=300)
    plt.close()
    print('Histogram saved to ' + plot4_path)

if __name__ == '__main__':
    main()