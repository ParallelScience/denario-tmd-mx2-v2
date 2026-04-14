# filename: codebase/step_4.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import shap
from datetime import datetime

if __name__ == '__main__':
    plt.rcParams['text.usetex'] = False
    data_dir = 'data'
    data_path = os.path.join(data_dir, 'processed_tmd_data.csv')
    df = pd.read_csv(data_path)
    model_path = os.path.join(data_dir, 'tmd_model.joblib')
    model_dict = joblib.load(model_path)
    rf_reg = model_dict['model_reg']
    features = model_dict['features']
    dos_mean_full = model_dict['dos_mean_full']
    df_imputed = df.copy()
    df_imputed['dos_at_fermi'] = df_imputed['dos_at_fermi'].fillna(dos_mean_full)
    if df_imputed['c_a_ratio'].isna().sum() > 0:
        df_imputed['c_a_ratio'] = df_imputed['c_a_ratio'].fillna(df_imputed['c_a_ratio'].mean())
    X = df_imputed[features]
    print('Calculating SHAP values for the RandomForestRegressor model...')
    explainer = shap.TreeExplainer(rf_reg)
    shap_values = explainer.shap_values(X)
    feature_names_with_units = []
    for f in features:
        if f == 'dos_at_fermi':
            feature_names_with_units.append('dos_at_fermi (states/eV)')
        elif f == 'd_band_filling':
            feature_names_with_units.append('d_band_filling (fraction)')
        elif f == 'en_difference':
            feature_names_with_units.append('en_difference (Pauling)')
        elif f == 'c_a_ratio':
            feature_names_with_units.append('c_a_ratio (dimensionless)')
        else:
            feature_names_with_units.append(f)
    X_plot = X.copy()
    X_plot.columns = feature_names_with_units
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_plot, show=False)
    plt.tight_layout()
    summary_plot_path = os.path.join(data_dir, 'shap_summary_1_' + timestamp + '.png')
    plt.savefig(summary_plot_path, dpi=300)
    plt.close()
    print('SHAP summary plot saved to ' + summary_plot_path)
    target_features_plot = ['d_band_filling (fraction)', 'en_difference (Pauling)', 'dos_at_fermi (states/eV)']
    for i, feat in enumerate(target_features_plot):
        plt.figure(figsize=(8, 6))
        shap.dependence_plot(feat, shap_values, X_plot, show=False, interaction_index=None, ax=plt.gca())
        plt.tight_layout()
        clean_feat_name = feat.split(' ')[0]
        dep_plot_path = os.path.join(data_dir, 'shap_dependence_' + clean_feat_name + '_' + str(i+2) + '_' + timestamp + '.png')
        plt.savefig(dep_plot_path, dpi=300)
        plt.close()
        print('SHAP dependence plot for ' + clean_feat_name + ' saved to ' + dep_plot_path)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_importance = pd.DataFrame({'feature': features, 'mean_abs_shap': mean_abs_shap}).sort_values('mean_abs_shap', ascending=False)
    print('\nMean Absolute SHAP Values (Feature Importance):')
    print(shap_importance.to_string(index=False))