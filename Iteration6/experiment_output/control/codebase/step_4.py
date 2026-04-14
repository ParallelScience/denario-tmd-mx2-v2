# filename: codebase/step_4.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from datetime import datetime
import shap

plt.rcParams['text.usetex'] = False

def main():
    data_dir = 'data'
    data_path = os.path.join(data_dir, 'processed_tmd_data.csv')
    df = pd.read_csv(data_path)
    model_path = os.path.join(data_dir, 'tmd_model.joblib')
    model_dict = joblib.load(model_path)
    if model_dict['type'] == 'rf':
        model = model_dict['reg']
        features = model_dict['features']
    else:
        return
    dos_mean = model_dict['dos_mean']
    df_full = df.copy()
    df_full['dos_at_fermi'] = df_full['dos_at_fermi'].fillna(dos_mean)
    if 'M_group_bin' not in df_full.columns:
        df_full['M_group_bin'] = (df_full['M_group'] >= 8).astype(int)
    X_full = df_full[features]
    mask_reg = df_full['pugh_ratio'].notna()
    X_background = X_full[mask_reg]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_background)
    if isinstance(shap_values, list):
        shap_values_array = shap_values[0]
    else:
        shap_values_array = shap_values
    timestamp = str(int(datetime.now().timestamp()))
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values_array, X_background, show=False)
    plt.title('SHAP Summary Plot for Pugh\'s Ratio')
    plt.tight_layout()
    summary_plot_path = os.path.join(data_dir, 'shap_summary_' + timestamp + '.png')
    plt.savefig(summary_plot_path, dpi=300)
    plt.close()
    print('SHAP summary plot saved to: ' + summary_plot_path)
    fig, ax = plt.subplots(figsize=(8, 6))
    shap.dependence_plot('d_band_filling', shap_values_array, X_background, interaction_index='dos_at_fermi', show=False, ax=ax)
    ax.set_title('SHAP Dependence: d_band_filling vs dos_at_fermi')
    plt.tight_layout()
    dependence_plot_path = os.path.join(data_dir, 'shap_dependence_' + timestamp + '.png')
    plt.savefig(dependence_plot_path, dpi=300)
    plt.close()
    print('SHAP dependence plot saved to: ' + dependence_plot_path)
    mean_abs_shap = np.abs(shap_values_array).mean(axis=0)
    feature_importance = pd.DataFrame({'feature': features, 'mean_abs_shap': mean_abs_shap})
    feature_importance = feature_importance.sort_values(by='mean_abs_shap', ascending=False).reset_index(drop=True)
    print('\nTop-5 features by mean absolute SHAP value:')
    print('-' * 60)
    for i in range(5):
        row = feature_importance.iloc[i]
        print(str(i+1) + '. ' + row['feature'] + ' (Mean |SHAP|: ' + str(round(row['mean_abs_shap'], 4)) + ')')
    print('-' * 60)

if __name__ == '__main__':
    main()