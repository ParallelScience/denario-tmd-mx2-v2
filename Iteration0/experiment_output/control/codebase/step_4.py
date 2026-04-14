# filename: codebase/step_4.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np
import json
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from datetime import datetime

os.environ['OMP_NUM_THREADS'] = '2'
plt.rcParams['text.usetex'] = False

if __name__ == '__main__':
    data_dir = 'data/'
    df_path = os.path.join(data_dir, 'processed_tmd_data_with_target.csv')
    df = pd.read_csv(df_path)
    df_elastic = df.dropna(subset=['is_robust']).copy()
    results_path = os.path.join(data_dir, 'model_training_results.json')
    with open(results_path, 'r') as f:
        results = json.load(f)
    features = results['features']
    best_params = results['best_hyperparameters']
    X = df_elastic[features]
    y = df_elastic['is_robust'].astype(int)
    pipeline = Pipeline([
        ('imputer', KNNImputer(n_neighbors=5)),
        ('rf', RandomForestClassifier(class_weight='balanced', random_state=42))
    ])
    pipeline.set_params(**best_params)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
    importances_list = []
    print('Calculating permutation feature importance on held-out folds...')
    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        pipeline.fit(X_train, y_train)
        result = permutation_importance(
            pipeline, X_test, y_test, n_repeats=5, random_state=42, scoring='average_precision', n_jobs=1
        )
        importances_list.append(result.importances_mean)
    importances_array = np.array(importances_list)
    mean_importances = np.mean(importances_array, axis=0)
    std_importances = np.std(importances_array, axis=0)
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance_Mean': mean_importances,
        'Importance_Std': std_importances
    })
    importance_df = importance_df.sort_values(by='Importance_Mean', ascending=False).reset_index(drop=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    importance_csv_path = os.path.join(data_dir, 'feature_importance_rankings_' + timestamp + '.csv')
    importance_df.to_csv(importance_csv_path, index=False)
    print('Feature importance rankings saved to ' + importance_csv_path)
    print('\nTop 10 Features by Permutation Importance (AUPRC):')
    print(importance_df.head(10).to_string(index=False))
    plt.figure(figsize=(12, 10))
    importance_df_sorted = importance_df.sort_values(by='Importance_Mean', ascending=True)
    plt.barh(
        importance_df_sorted['Feature'],
        importance_df_sorted['Importance_Mean'],
        xerr=importance_df_sorted['Importance_Std'],
        capsize=4,
        color='steelblue'
    )
    plt.xlabel('Permutation Importance (Mean AUPRC Decrease)')
    plt.title('Feature Importance on Held-out Folds')
    plt.tight_layout()
    importance_plot_path = os.path.join(data_dir, 'feature_importance_bar_chart_1_' + timestamp + '.png')
    plt.savefig(importance_plot_path, dpi=300)
    print('Feature importance bar chart saved to ' + importance_plot_path)
    plt.close()
    model_path = os.path.join(data_dir, 'rf_model.joblib')
    full_model = joblib.load(model_path)
    X_imputed_array = full_model.named_steps['imputer'].transform(X)
    X_imputed = pd.DataFrame(X_imputed_array, columns=X.columns, index=X.index)
    pdp_features = ['d_band_filling', 'en_difference', 'dos_at_fermi']
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    print('\nGenerating Partial Dependence Plots...')
    PartialDependenceDisplay.from_estimator(
        full_model, X_imputed, features=pdp_features, ax=ax, grid_resolution=50, random_state=42
    )
    fig.suptitle('Partial Dependence Plots for Top Electronic Features', fontsize=16)
    fig.tight_layout()
    pdp_plot_path = os.path.join(data_dir, 'partial_dependence_plots_2_' + timestamp + '.png')
    fig.savefig(pdp_plot_path, dpi=300)
    print('Partial Dependence Plots saved to ' + pdp_plot_path)
    plt.close(fig)