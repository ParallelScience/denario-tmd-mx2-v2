# filename: codebase/step_4.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from sklearn.neighbors import KNeighborsRegressor

def main():
    data_dir = 'data'
    df = pd.read_csv(os.path.join(data_dir, 'elastic_subset.csv'))
    with open(os.path.join(data_dir, 'lomo_cv_indices.json'), 'r') as f:
        lomo_cv = json.load(f)
    with open(os.path.join(data_dir, 'lomo_cv_results.json'), 'r') as f:
        lomo_results = json.load(f)
    ohe_features = [c for c in df.columns if c.startswith('magnetic_ordering_') or c.startswith('phase_')]
    rf_features = ['d_band_filling', 'M_soc_proxy', 'en_difference', 'M_atomic_radius', 'X_atomic_radius', 'dos_at_fermi', 'crystal_system_ord'] + ohe_features
    knn_features = ['band_gap', 'd_band_filling', 'en_difference', 'M_soc_proxy']
    sens_features = rf_features + ['volume_per_atom']
    fold_importances = []
    y_true_all = []
    y_prob_base_all = []
    y_prob_sens_all = []
    sens_results = {}
    for metal, indices in lomo_cv.items():
        train_idx = indices['train']
        test_idx = indices['test']
        df_train = df.iloc[train_idx].copy()
        df_test = df.iloc[test_idx].copy()
        valid_train = df_train[~df_train['dos_at_fermi_imputed_flag']]
        if len(valid_train) > 0:
            knn = KNeighborsRegressor(n_neighbors=min(5, len(valid_train)))
            knn.fit(valid_train[knn_features], valid_train['dos_at_fermi'])
            impute_train_mask = df_train['dos_at_fermi_imputed_flag']
            if impute_train_mask.sum() > 0:
                df_train.loc[impute_train_mask, 'dos_at_fermi'] = knn.predict(df_train.loc[impute_train_mask, knn_features])
            impute_test_mask = df_test['dos_at_fermi_imputed_flag']
            if impute_test_mask.sum() > 0:
                df_test.loc[impute_test_mask, 'dos_at_fermi'] = knn.predict(df_test.loc[impute_test_mask, knn_features])
        else:
            df_train.loc[df_train['dos_at_fermi_imputed_flag'], 'dos_at_fermi'] = 0.0
            df_test.loc[df_test['dos_at_fermi_imputed_flag'], 'dos_at_fermi'] = 0.0
        best_hp = lomo_results[metal]['best_hp']
        rf_base = RandomForestClassifier(n_estimators=best_hp['n_estimators'], max_depth=best_hp['max_depth'], min_samples_split=best_hp['min_samples_split'], class_weight='balanced', random_state=42, n_jobs=1)
        rf_base.fit(df_train[rf_features], df_train['is_robust'])
        y_prob_base = rf_base.predict_proba(df_test[rf_features])[:, 1]
        rf_sens = RandomForestClassifier(n_estimators=best_hp['n_estimators'], max_depth=best_hp['max_depth'], min_samples_split=best_hp['min_samples_split'], class_weight='balanced', random_state=42, n_jobs=1)
        rf_sens.fit(df_train[sens_features], df_train['is_robust'])
        y_prob_sens = rf_sens.predict_proba(df_test[sens_features])[:, 1]
        y_test = df_test['is_robust'].values
        y_true_all.extend(y_test)
        y_prob_base_all.extend(y_prob_base)
        y_prob_sens_all.extend(y_prob_sens)
        if len(np.unique(y_test)) > 1:
            auprc_base = average_precision_score(y_test, y_prob_base)
            auprc_sens = average_precision_score(y_test, y_prob_sens)
            diff = auprc_sens - auprc_base
            n_repeats = 10
            rng = np.random.RandomState(42)
            X_test_base = df_test[rf_features].copy()
            importances = np.zeros((len(rf_features), n_repeats))
            for col_idx, col_name in enumerate(rf_features):
                for n in range(n_repeats):
                    X_perm = X_test_base.copy()
                    X_perm[col_name] = rng.permutation(X_perm[col_name].values)
                    score = average_precision_score(y_test, rf_base.predict_proba(X_perm)[:, 1])
                    importances[col_idx, n] = auprc_base - score
            fold_importances.append(importances.mean(axis=1))
        else:
            auprc_base = np.nan
            auprc_sens = np.nan
            diff = np.nan
        sens_results[metal] = {'auprc_base': auprc_base, 'auprc_sens': auprc_sens, 'diff': diff}
    fold_importances = np.array(fold_importances)
    mean_importances = fold_importances.mean(axis=0)
    std_importances = fold_importances.std(axis=0)
    imp_df = pd.DataFrame({'feature': rf_features, 'importance_mean': mean_importances, 'importance_std': std_importances}).sort_values('importance_mean', ascending=False).reset_index(drop=True)
    print('--- Permutation Feature Importance (AUPRC) ---')
    for _, row in imp_df.iterrows():
        print(row['feature'] + ': ' + str(round(row['importance_mean'], 4)) + ' +/- ' + str(round(row['importance_std'], 4)))
    print('----------------------------------------------\n')
    pooled_auprc_base = average_precision_score(y_true_all, y_prob_base_all)
    pooled_auprc_sens = average_precision_score(y_true_all, y_prob_sens_all)
    pooled_diff = pooled_auprc_sens - pooled_auprc_base
    print('--- Sensitivity Analysis (Adding volume_per_atom) ---')
    for metal in sorted(sens_results.keys()):
        res = sens_results[metal]
        if not np.isnan(res['auprc_base']):
            print('Metal ' + metal + ': Base AUPRC = ' + str(round(res['auprc_base'], 4)) + ', Sens AUPRC = ' + str(round(res['auprc_sens'], 4)) + ', Diff = ' + str(round(res['diff'], 4)))
        else:
            print('Metal ' + metal + ': NaN (single class in test set)')
    print('\nGlobal Pooled Base AUPRC: ' + str(round(pooled_auprc_base, 4)))
    print('Global Pooled Sens AUPRC: ' + str(round(pooled_auprc_sens, 4)))
    print('Global Pooled Difference: ' + str(round(pooled_diff, 4)))
    print('-----------------------------------------------------\n')
    imp_df.to_csv(os.path.join(data_dir, 'feature_importance.csv'), index=False)
    print('Feature importances saved to ' + os.path.join(data_dir, 'feature_importance.csv'))
    with open(os.path.join(data_dir, 'sensitivity_results.json'), 'w') as f:
        json.dump({'per_fold': {m: {k: (float(v) if not np.isnan(v) else None) for k, v in res.items()} for m, res in sens_results.items()}, 'pooled_base_auprc': float(pooled_auprc_base), 'pooled_sens_auprc': float(pooled_auprc_sens), 'pooled_diff': float(pooled_diff)}, f, indent=4)
    print('Sensitivity results saved to ' + os.path.join(data_dir, 'sensitivity_results.json'))

if __name__ == '__main__':
    main()