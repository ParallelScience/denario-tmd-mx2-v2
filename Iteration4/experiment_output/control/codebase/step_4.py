# filename: codebase/step_4.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import shap
import joblib
from scipy.stats import kruskal

mpl.rcParams['text.usetex'] = False

if __name__ == '__main__':
    data_dir = 'data/'
    df = pd.read_csv(os.path.join(data_dir, 'training_dataset_step2.csv'))
    with open(os.path.join(data_dir, 'selected_features.txt'), 'r') as f:
        selected_features = [line.strip() for line in f.readlines() if line.strip()]
    X = df[selected_features]
    model = joblib.load(os.path.join(data_dir, 'rf_model_step3.joblib'))
    print('1. Kruskal-Wallis test for M_soc_proxy vs magnetic_ordering:')
    if 'magnetic_ordering' in df.columns and 'M_soc_proxy' in df.columns:
        mo_groups = df['magnetic_ordering'].dropna().unique()
        groups_data = [df[df['magnetic_ordering'] == mo]['M_soc_proxy'].dropna().values for mo in mo_groups]
        groups_data = [g for g in groups_data if len(g) > 0]
        if len(groups_data) > 1:
            stat, pval = kruskal(*groups_data)
            print('  - Magnetic orderings present: ' + ', '.join([str(mo) for mo in mo_groups]))
            print('  - Kruskal-Wallis H-statistic: ' + str(round(stat, 4)))
            print('  - p-value: ' + str(pval))
        else:
            print('  - Not enough groups for Kruskal-Wallis test.')
    else:
        print('  - Required columns not found in the dataset.')
    print('\n2. SHAP Analysis:')
    explainer = shap.TreeExplainer(model)
    shap_values_all = explainer.shap_values(X)
    if isinstance(shap_values_all, list):
        shap_values = shap_values_all[1]
    elif len(np.shape(shap_values_all)) == 3:
        shap_values = shap_values_all[:, :, 1]
    else:
        shap_values = shap_values_all
    timestamp = int(time.time())
    shap.summary_plot(shap_values, X, show=False)
    fig = plt.gcf()
    fig.suptitle('SHAP Summary Plot for Mechanical Viability', fontsize=14, y=1.02)
    summary_plot_filename = os.path.join(data_dir, 'shap_summary_4_' + str(timestamp) + '.png')
    plt.savefig(summary_plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print('SHAP summary plot saved to ' + summary_plot_filename)
    dos_col = 'dos_at_fermi_imputed' if 'dos_at_fermi_imputed' in df.columns else 'dos_at_fermi'
    if 'd_band_filling' in X.columns:
        if dos_col in X.columns:
            shap.dependence_plot('d_band_filling', shap_values, X, interaction_index=dos_col, show=False)
        else:
            print('  - Warning: ' + dos_col + ' is not in selected features. Plotting without interaction color.')
            shap.dependence_plot('d_band_filling', shap_values, X, interaction_index=None, show=False)
        fig = plt.gcf()
        fig.suptitle('SHAP Dependence: d_band_filling vs ' + dos_col, fontsize=14, y=1.02)
        dep_plot_filename = os.path.join(data_dir, 'shap_dependence_4_' + str(timestamp) + '.png')
        plt.savefig(dep_plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print('SHAP dependence plot saved to ' + dep_plot_filename)
    else:
        print('  - d_band_filling is not in the selected features. Cannot generate its SHAP dependence plot.')