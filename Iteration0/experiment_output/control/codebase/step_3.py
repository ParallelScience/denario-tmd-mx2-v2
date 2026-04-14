# filename: codebase/step_3.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np
import json
import joblib
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline

os.environ['OMP_NUM_THREADS'] = '2'

if __name__ == '__main__':
    data_dir = 'data/'
    df_path = os.path.join(data_dir, 'processed_tmd_data_with_target.csv')
    df = pd.read_csv(df_path)
    df_elastic = df.dropna(subset=['is_robust']).copy()
    base_features = ['band_gap', 'M_val', 'X_period', 'crystal_system', 'c_a_ratio', 'volume_per_atom', 'energy_above_hull', 'd_band_filling', 'en_difference', 'dos_at_fermi', 'M_soc_proxy']
    one_hot_features = [col for col in df.columns if col.startswith('magnetic_ordering_') or (col.startswith('phase_') and col not in ['phase', 'phase_median_G_vrh'])]
    features = base_features + one_hot_features
    X = df_elastic[features]
    y = df_elastic['is_robust'].astype(int)
    pipeline = Pipeline([('imputer', KNNImputer(n_neighbors=5)), ('rf', RandomForestClassifier(class_weight='balanced', random_state=42))])
    param_grid = {'rf__n_estimators': [50, 100, 200], 'rf__max_depth': [None, 5, 10], 'rf__min_samples_split': [2, 5, 10]}
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
    scoring = {'AUPRC': 'average_precision', 'ROC_AUC': 'roc_auc', 'Balanced_Accuracy': 'balanced_accuracy'}
    grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=cv, scoring=scoring, refit='AUPRC', n_jobs=8, return_train_score=False)
    grid_search.fit(X, y)
    best_params = grid_search.best_params_
    best_index = grid_search.best_index_
    results = grid_search.cv_results_
    mean_auprc = results['mean_test_AUPRC'][best_index]
    std_auprc = results['std_test_AUPRC'][best_index]
    mean_roc_auc = results['mean_test_ROC_AUC'][best_index]
    std_roc_auc = results['std_test_ROC_AUC'][best_index]
    mean_bal_acc = results['mean_test_Balanced_Accuracy'][best_index]
    std_bal_acc = results['std_test_Balanced_Accuracy'][best_index]
    model_path = os.path.join(data_dir, 'rf_model.joblib')
    joblib.dump(grid_search.best_estimator_, model_path)
    output_results = {'best_hyperparameters': best_params, 'metrics': {'AUPRC': {'mean': mean_auprc, 'std': std_auprc}, 'ROC_AUC': {'mean': mean_roc_auc, 'std': std_roc_auc}, 'Balanced_Accuracy': {'mean': mean_bal_acc, 'std': std_bal_acc}}, 'features': features}
    results_path = os.path.join(data_dir, 'model_training_results.json')
    with open(results_path, 'w') as f:
        json.dump(output_results, f, indent=4)