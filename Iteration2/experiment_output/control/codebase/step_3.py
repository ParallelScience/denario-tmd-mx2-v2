# filename: codebase/step_3.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import average_precision_score, precision_recall_curve, auc
import joblib
import time

os.environ['OMP_NUM_THREADS'] = '2'

def get_stratified_loco_folds(df, group_col, target_col):
    groups = df[group_col].values
    y = df[target_col].values
    unique_groups = np.unique(groups)
    group_stats = []
    for g in unique_groups:
        idx = np.where(groups == g)[0]
        pos = y[idx].sum()
        neg = len(idx) - pos
        group_stats.append({'group': g, 'pos': pos, 'neg': neg, 'idx': idx})
    folds_indices = []
    current_pos = 0
    current_neg = 0
    current_idx = []
    for stat in group_stats:
        current_idx.extend(stat['idx'])
        current_pos += stat['pos']
        current_neg += stat['neg']
        if current_pos >= 1 and current_neg >= 1:
            folds_indices.append(current_idx)
            current_idx = []
            current_pos = 0
            current_neg = 0
    if len(current_idx) > 0:
        if len(folds_indices) > 0:
            folds_indices[-1].extend(current_idx)
        else:
            folds_indices.append(current_idx)
    cv_splits = []
    all_indices = np.arange(len(y))
    for test_idx in folds_indices:
        train_idx = np.setdiff1d(all_indices, test_idx)
        cv_splits.append((train_idx, test_idx))
    return cv_splits

if __name__ == '__main__':
    data_dir = 'data/'
    train_path = os.path.join(data_dir, 'train_dataset_step2.csv')
    train_df = pd.read_csv(train_path)
    baseline_features = ['X_atomic_radius', 'crystal_system_encoded', 'M_group']
    full_features = ['X_atomic_radius', 'crystal_system_encoded', 'M_group', 'd_band_filling', 'dos_at_fermi', 'M_soc_proxy', 'en_difference', 'magnetic_ordering_encoded', 'phase_encoded', 'imputation_uncertainty', 'is_dos_missing']
    X_base = train_df[baseline_features].values
    X_full = train_df[full_features].values
    y_class = train_df['is_robust'].values
    y_reg = train_df['is_stable_structure'].values
    cv_splits = get_stratified_loco_folds(train_df, 'M_group', 'is_robust')
    print('Created ' + str(len(cv_splits)) + ' stratified LOCO folds based on M_group.')
    param_grid = {'n_estimators': [50, 100, 200, 300], 'max_depth': [None, 3, 5, 7, 10], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}
    print('\nOptimizing Baseline Model (Static Descriptors)...')
    rf_base = RandomForestClassifier(class_weight='balanced', random_state=42)
    search_base = RandomizedSearchCV(rf_base, param_grid, n_iter=30, scoring='average_precision', cv=cv_splits, random_state=42, n_jobs=8)
    search_base.fit(X_base, y_class)
    best_base_model = search_base.best_estimator_
    print('Optimizing Full Model (Electronic Descriptors)...')
    rf_full = RandomForestClassifier(class_weight='balanced', random_state=42)
    search_full = RandomizedSearchCV(rf_full, param_grid, n_iter=30, scoring='average_precision', cv=cv_splits, random_state=42, n_jobs=8)
    search_full.fit(X_full, y_class)
    best_full_model = search_full.best_estimator_
    def evaluate_model(model, X, y, cv_splits):
        fold_auprc = []
        oof_preds = np.zeros(len(y))
        for train_idx, test_idx in cv_splits:
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]
            model.fit(X_train, y_train)
            preds = model.predict_proba(X_test)[:, 1]
            oof_preds[test_idx] = preds
            if len(np.unique(y_test)) > 1:
                score = average_precision_score(y_test, preds)
                fold_auprc.append(score)
            else:
                fold_auprc.append(np.nan)
        return fold_auprc, oof_preds
    base_fold_auprc, base_oof_preds = evaluate_model(best_base_model, X_base, y_class, cv_splits)
    full_fold_auprc, full_oof_preds = evaluate_model(best_full_model, X_full, y_class, cv_splits)
    print('\n--- Baseline Model Performance ---')
    print('Best params: ' + str(search_base.best_params_))
    for i, score in enumerate(base_fold_auprc):
        print('  Fold ' + str(i+1) + ' AUPRC: ' + str(round(score, 4)))
    print('  Mean AUPRC: ' + str(round(np.nanmean(base_fold_auprc), 4)))
    print('\n--- Full Model Performance ---')
    print('Best params: ' + str(search_full.best_params_))
    for i, score in enumerate(full_fold_auprc):
        print('  Fold ' + str(i+1) + ' AUPRC: ' + str(round(score, 4)))
    print('  Mean AUPRC: ' + str(round(np.nanmean(full_fold_auprc), 4)))
    print('\nTraining Full Regression Model for elastic_anisotropy...')
    rf_reg = RandomForestRegressor(random_state=42)
    search_reg = RandomizedSearchCV(rf_reg, param_grid, n_iter=30, scoring='neg_mean_squared_error', cv=cv_splits, random_state=42, n_jobs=8)
    search_reg.fit(X_full, y_reg)
    best_reg_model = search_reg.best_estimator_
    best_base_model.fit(X_base, y_class)
    best_full_model.fit(X_full, y_class)
    best_reg_model.fit(X_full, y_reg)
    joblib.dump(best_base_model, os.path.join(data_dir, 'baseline_rf_classifier.joblib'))
    joblib.dump(best_full_model, os.path.join(data_dir, 'full_rf_classifier.joblib'))
    joblib.dump(best_reg_model, os.path.join(data_dir, 'full_rf_regressor.joblib'))
    base_precision, base_recall, _ = precision_recall_curve(y_class, base_oof_preds)
    full_precision, full_recall, _ = precision_recall_curve(y_class, full_oof_preds)
    base_auc = auc(base_recall, base_precision)
    full_auc = auc(full_recall, full_precision)
    plt.rcParams['text.usetex'] = False
    plt.figure(figsize=(8, 6))
    plt.plot(base_recall, base_precision, label='Baseline Model (AUPRC = ' + str(round(base_auc, 3)) + ')', lw=2)
    plt.plot(full_recall, full_precision, label='Full Model (AUPRC = ' + str(round(full_auc, 3)) + ')', lw=2)
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve Comparison (LOCO CV)', fontsize=14)
    plt.legend(loc='best', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    timestamp = int(time.time())
    plot_filename = 'pr_curve_comparison_1_' + str(timestamp) + '.png'
    plot_filepath = os.path.join(data_dir, plot_filename)
    plt.savefig(plot_filepath, dpi=300)
    print('\nPrecision-Recall curve plot saved to ' + plot_filepath)
    cv_results_df = pd.DataFrame({'fold': range(1, len(cv_splits) + 1), 'baseline_auprc': base_fold_auprc, 'full_auprc': full_fold_auprc})
    cv_results_path = os.path.join(data_dir, 'cv_results_step3.csv')
    cv_results_df.to_csv(cv_results_path, index=False)
    print('Cross-validation results saved to ' + cv_results_path)