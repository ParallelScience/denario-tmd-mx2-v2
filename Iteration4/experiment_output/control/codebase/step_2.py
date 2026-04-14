# filename: codebase/step_2.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import time
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import mannwhitneyu
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.feature_selection import RFECV, RFE
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

mpl.rcParams['text.usetex'] = False

if __name__ == '__main__':
    full_data_path = '/home/node/work/projects/materials_project_v1/tmd_data_enriched.csv'
    full_df = pd.read_csv(full_data_path)
    dos_missing = full_df[full_df['dos_at_fermi'].isna()]
    dos_present = full_df[full_df['dos_at_fermi'].notna()]
    print('1. Mann-Whitney U test for missing dos_at_fermi:')
    for col in ['energy_above_hull', 'volume_per_atom']:
        stat, pval = mannwhitneyu(dos_missing[col].dropna(), dos_present[col].dropna(), alternative='two-sided')
        print('  - ' + col + ': U-statistic = ' + str(round(stat, 4)) + ', p-value = ' + str(pval))
    df_elastic = pd.read_csv('data/processed_tmd_data.csv')
    p5_dos = df_elastic['dos_at_fermi'].quantile(0.05)
    df_elastic['is_dos_missing'] = df_elastic['dos_at_fermi'].isna().astype(int)
    df_elastic['dos_at_fermi_imputed'] = df_elastic['dos_at_fermi'].fillna(p5_dos)
    base_features = ['volume_per_atom', 'c_a_ratio', 'band_gap', 'efermi', 'dos_at_fermi_imputed', 'total_magnetization', 'M_val', 'M_Z', 'M_en', 'M_ie1', 'M_atomic_radius', 'M_group', 'M_period', 'M_soc_proxy', 'X_en', 'X_ie1', 'X_atomic_radius', 'X_period', 'en_difference', 'bond_radius_sum', 'd_count_m4plus', 'd_band_filling']
    for col in base_features:
        if df_elastic[col].isna().any():
            df_elastic[col] = df_elastic[col].fillna(df_elastic[col].median())
    mag_dummies = pd.get_dummies(df_elastic['magnetic_ordering'], prefix='mag').astype(int)
    df_elastic = pd.concat([df_elastic, mag_dummies], axis=1)
    mag_cols = mag_dummies.columns.tolist()
    X_with_indicator = df_elastic[base_features + mag_cols + ['is_dos_missing']]
    X_without_indicator = df_elastic[base_features + mag_cols]
    y = df_elastic['is_viable'].astype(int)
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {'roc_auc': 'roc_auc', 'balanced_accuracy': 'balanced_accuracy'}
    scores_with = cross_validate(rf, X_with_indicator, y, cv=cv, scoring=scoring)
    scores_without = cross_validate(rf, X_without_indicator, y, cv=cv, scoring=scoring)
    print('\n2. Model Performance Comparison (Imputation with vs without is_dos_missing):')
    print('  With indicator: ROC-AUC: ' + str(round(scores_with['test_roc_auc'].mean(), 4)))
    print('  Without indicator: ROC-AUC: ' + str(round(scores_without['test_roc_auc'].mean(), 4)))
    if scores_with['test_roc_auc'].mean() >= scores_without['test_roc_auc'].mean():
        print('  -> Retaining is_dos_missing indicator.')
        final_features = base_features + mag_cols + ['is_dos_missing']
    else:
        print('  -> Dropping is_dos_missing indicator.')
        final_features = base_features + mag_cols
    rf.fit(X_with_indicator, y)
    idx = list(X_with_indicator.columns).index('is_dos_missing')
    imp = rf.feature_importances_[idx]
    print('  -> Feature importance of is_dos_missing: ' + str(round(imp, 4)))
    print('\n3. Variance Inflation Factor (VIF) for continuous features:')
    X_cont = df_elastic[base_features].copy()
    X_cont_const = add_constant(X_cont)
    vifs = []
    features_vif = []
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for i, col in enumerate(X_cont_const.columns):
            if col == 'const': continue
            try: vif = variance_inflation_factor(X_cont_const.values, i)
            except Exception: vif = np.inf
            vifs.append(vif)
            features_vif.append(col)
    vif_data = pd.DataFrame({'feature': features_vif, 'VIF': vifs})
    high_vif = vif_data[vif_data['VIF'] > 10].sort_values(by='VIF', ascending=False)
    print('  Features with VIF > 10:')
    for _, row in high_vif.iterrows():
        print('    - ' + row['feature'] + ': ' + str(round(row['VIF'], 2)))
    print('\n4. Performing Recursive Feature Elimination (RFECV)...')
    X_final = df_elastic[final_features]
    rf_rfe = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rfecv = RFECV(estimator=rf_rfe, step=1, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), scoring='roc_auc', min_features_to_select=1, n_jobs=-1)
    rfecv.fit(X_final, y)
    mean_scores = rfecv.cv_results_['mean_test_score']
    n_features_grid = np.arange(rfecv.min_features_to_select, len(X_final.columns) + 1)
    valid_indices = np.where((n_features_grid >= 6) & (n_features_grid <= 10))[0]
    if len(valid_indices) > 0:
        best_idx_in_range = valid_indices[np.argmax(mean_scores[valid_indices])]
        target_n_features = n_features_grid[best_idx_in_range]
    else: target_n_features = rfecv.n_features_
    print('  Targeting 6-10 features, selected number of features: ' + str(target_n_features))
    rfe_final = RFE(estimator=rf_rfe, n_features_to_select=target_n_features, step=1)
    rfe_final.fit(X_final, y)
    selected_features = list(X_final.columns[rfe_final.support_])
    plt.figure(figsize=(10, 6))
    plt.plot(n_features_grid, mean_scores, marker='o', label='CV ROC-AUC')
    plt.axvline(target_n_features, color='red', linestyle='--', label='Selected')
    plt.title('RFECV: ROC-AUC vs. Number of Features')
    plt.xlabel('Number of Features Selected')
    plt.ylabel('Cross-Validation ROC-AUC')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plot_filename = os.path.join('data', 'rfecv_scores_2_' + str(int(time.time())) + '.png')
    plt.savefig(plot_filename, dpi=300)
    print('  Plot saved to ' + plot_filename)
    df_elastic.to_csv('data/training_dataset_step2.csv', index=False)
    with open('data/selected_features.txt', 'w') as f:
        for feature in selected_features: f.write(feature + '\n')
    print('\n5. Data Saving: Selected features and training dataset saved.')