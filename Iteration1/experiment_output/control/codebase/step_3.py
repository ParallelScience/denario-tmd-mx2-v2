# filename: codebase/step_3.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from sklearn.neighbors import KNeighborsRegressor
from itertools import product
from joblib import Parallel, delayed

os.environ['OMP_NUM_THREADS'] = '2'

def evaluate_hp(hp, df_outer_train, inner_metals, knn_features, rf_features):
    y_inner_true_all = []
    y_inner_prob_all = []
    for inner_metal in inner_metals:
        inner_train_mask = df_outer_train['metal'] != inner_metal
        inner_val_mask = df_outer_train['metal'] == inner_metal
        df_inner_train = df_outer_train[inner_train_mask].copy()
        df_inner_val = df_outer_train[inner_val_mask].copy()
        valid_inner_train = df_inner_train[~df_inner_train['dos_at_fermi_imputed_flag']]
        if len(valid_inner_train) > 0:
            knn = KNeighborsRegressor(n_neighbors=min(5, len(valid_inner_train)))
            knn.fit(valid_inner_train[knn_features], valid_inner_train['dos_at_fermi'])
            impute_train_mask = df_inner_train['dos_at_fermi_imputed_flag']
            if impute_train_mask.sum() > 0:
                df_inner_train.loc[impute_train_mask, 'dos_at_fermi'] = knn.predict(df_inner_train.loc[impute_train_mask, knn_features])
            impute_val_mask = df_inner_val['dos_at_fermi_imputed_flag']
            if impute_val_mask.sum() > 0:
                df_inner_val.loc[impute_val_mask, 'dos_at_fermi'] = knn.predict(df_inner_val.loc[impute_val_mask, knn_features])
        else:
            df_inner_train.loc[df_inner_train['dos_at_fermi_imputed_flag'], 'dos_at_fermi'] = 0.0
            df_inner_val.loc[df_inner_val['dos_at_fermi_imputed_flag'], 'dos_at_fermi'] = 0.0
        rf = RandomForestClassifier(n_estimators=hp['n_estimators'], max_depth=hp['max_depth'], min_samples_split=hp['min_samples_split'], class_weight='balanced', random_state=42, n_jobs=1)
        X_inner_train = df_inner_train[rf_features]
        y_inner_train = df_inner_train['is_robust']
        X_inner_val = df_inner_val[rf_features]
        y_inner_val = df_inner_val['is_robust']
        if len(y_inner_train.unique()) < 2:
            continue
        rf.fit(X_inner_train, y_inner_train)
        y_prob = rf.predict_proba(X_inner_val)[:, 1]
        y_inner_true_all.extend(y_inner_val.tolist())
        y_inner_prob_all.extend(y_prob.tolist())
    if len(np.unique(y_inner_true_all)) > 1:
        inner_auprc = average_precision_score(y_inner_true_all, y_inner_prob_all)
    else:
        inner_auprc = 0.0
    return hp, inner_auprc

def main():
    data_dir = 'data'
    df = pd.read_csv(os.path.join(data_dir, 'elastic_subset.csv'))
    with open(os.path.join(data_dir, 'lomo_cv_indices.json'), 'r') as f:
        lomo_cv = json.load(f)
    ohe_features = [c for c in df.columns if c.startswith('magnetic_ordering_') or c.startswith('phase_')]
    rf_features = ['d_band_filling', 'M_soc_proxy', 'en_difference', 'M_atomic_radius', 'X_atomic_radius', 'dos_at_fermi', 'crystal_system_ord'] + ohe_features
    knn_features = ['band_gap', 'd_band_filling', 'en_difference', 'M_soc_proxy']
    print('Features used for Random Forest:')
    for f in rf_features:
        print('  - ' + f)
    print('Total features: ' + str(len(rf_features)) + '\n')
    param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 10, None], 'min_samples_split': [2, 5, 10]}
    keys, values = zip(*param_grid.items())
    hyperparams = [dict(zip(keys, v)) for v in product(*values)]
    results = {}
    print('Starting Nested LOMO CV for Hyperparameter Optimization...')
    for outer_metal, indices in lomo_cv.items():
        outer_train_idx = indices['train']
        outer_test_idx = indices['test']
        df_outer_train = df.iloc[outer_train_idx].copy()
        df_outer_test = df.iloc[outer_test_idx].copy()
        inner_metals = df_outer_train['metal'].unique()
        hp_results = Parallel(n_jobs=8)(delayed(evaluate_hp)(hp, df_outer_train, inner_metals, knn_features, rf_features) for hp in hyperparams)
        best_hp = None
        best_inner_auprc = -1.0
        for hp, auprc in hp_results:
            if auprc > best_inner_auprc:
                best_inner_auprc = auprc
                best_hp = hp
        valid_outer_train = df_outer_train[~df_outer_train['dos_at_fermi_imputed_flag']]
        if len(valid_outer_train) > 0:
            knn = KNeighborsRegressor(n_neighbors=min(5, len(valid_outer_train)))
            knn.fit(valid_outer_train[knn_features], valid_outer_train['dos_at_fermi'])
            impute_train_mask = df_outer_train['dos_at_fermi_imputed_flag']
            if impute_train_mask.sum() > 0:
                df_outer_train.loc[impute_train_mask, 'dos_at_fermi'] = knn.predict(df_outer_train.loc[impute_train_mask, knn_features])
            impute_test_mask = df_outer_test['dos_at_fermi_imputed_flag']
            if impute_test_mask.sum() > 0:
                df_outer_test.loc[impute_test_mask, 'dos_at_fermi'] = knn.predict(df_outer_test.loc[impute_test_mask, knn_features])
        else:
            df_outer_train.loc[df_outer_train['dos_at_fermi_imputed_flag'], 'dos_at_fermi'] = 0.0
            df_outer_test.loc[df_outer_test['dos_at_fermi_imputed_flag'], 'dos_at_fermi'] = 0.0
        rf_outer = RandomForestClassifier(n_estimators=best_hp['n_estimators'], max_depth=best_hp['max_depth'], min_samples_split=best_hp['min_samples_split'], class_weight='balanced', random_state=42, n_jobs=1)
        X_outer_train = df_outer_train[rf_features]
        y_outer_train = df_outer_train['is_robust']
        X_outer_test = df_outer_test[rf_features]
        y_outer_test = df_outer_test['is_robust']
        rf_outer.fit(X_outer_train, y_outer_train)
        y_prob_outer = rf_outer.predict_proba(X_outer_test)[:, 1]
        if len(y_outer_test.unique()) > 1:
            outer_auprc = average_precision_score(y_outer_test, y_prob_outer)
        else:
            outer_auprc = np.nan
        results[outer_metal] = {'best_hp': best_hp, 'inner_auprc': best_inner_auprc, 'outer_auprc': outer_auprc, 'y_true': y_outer_test.tolist(), 'y_prob': y_prob_outer.tolist()}
    print('\n--- Nested LOMO CV Results ---')
    all_y_true = []
    all_y_prob = []
    for metal, res in results.items():
        hp_str = 'n_est=' + str(res['best_hp']['n_estimators']) + ', max_depth=' + str(res['best_hp']['max_depth']) + ', min_samples_split=' + str(res['best_hp']['min_samples_split'])
        auprc_val = res['outer_auprc']
        auprc_str = str(round(auprc_val, 4)) if not np.isnan(auprc_val) else 'NaN (single class)'
        print('Metal ' + metal + ':')
        print('  Best HP: ' + hp_str)
        print('  Inner AUPRC: ' + str(round(res['inner_auprc'], 4)))
        print('  Outer AUPRC: ' + auprc_str)
        all_y_true.extend(res['y_true'])
        all_y_prob.extend(res['y_prob'])
    pooled_auprc = average_precision_score(all_y_true, all_y_prob)
    print('\n--- Global Performance ---')
    print('Pooled Outer AUPRC: ' + str(round(pooled_auprc, 4)))
    print('--------------------------\n')
    results_to_save = {}
    for metal, res in results.items():
        results_to_save[metal] = {'best_hp': res['best_hp'], 'inner_auprc': float(res['inner_auprc']), 'outer_auprc': float(res['outer_auprc']) if not np.isnan(res['outer_auprc']) else None, 'y_true': res['y_true'], 'y_prob': res['y_prob']}
    output_path = os.path.join(data_dir, 'lomo_cv_results.json')
    with open(output_path, 'w') as f:
        json.dump(results_to_save, f, indent=4)
    print('LOMO CV results saved to ' + output_path)

if __name__ == '__main__':
    main()