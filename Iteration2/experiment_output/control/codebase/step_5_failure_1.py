# filename: codebase/step_5.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import time
import shap
from sklearn.inspection import PartialDependenceDisplay

os.environ['OMP_NUM_THREADS'] = '2'

if __name__ == '__main__':
    data_dir = 'data/'
    train_path = os.path.join(data_dir, 'train_dataset_step2.csv')
    train_df = pd.read_csv(train_path)
    
    full_features = [
        'X_atomic_radius', 'crystal_system_encoded', 'M_group', 
        'd_band_filling', 'dos_at_fermi', 'M_soc_proxy', 
        'en_difference', 'magnetic_ordering_encoded', 
        'phase_encoded', 'imputation_uncertainty', 'is_dos_missing'
    ]
    
    X_full_df = train_df[full_features]
    
    model_path = os.path.join(data_dir, 'full_rf_classifier.joblib')
    best_full_model = joblib.load(model_path)
    
    print('Computing SHAP values...')
    explainer = shap.TreeExplainer(best_full_model)
    shap_values_obj = explainer.shap_values(X_full_df)
    
    if isinstance(shap_values_obj, list):
        shap_values_pos = shap_values_obj[1]
    elif len(np.array(shap_values_obj).shape) == 3:
        shap_values_pos = np.array(shap_values_obj)[:, :, 1]
    else:
        shap_values_pos = shap_values_obj
        
    mean_abs_shap = np.abs(shap_values_pos).mean(axis=0)
    shap_df = pd.DataFrame({'feature': full_features, 'mean_abs_shap': mean_abs_shap})
    shap_df = shap_df.sort_values('mean_abs_shap', ascending=False)
    
    print('\n--- Top-5 Most Important Features (Mean Absolute SHAP) ---')
    for i, row in shap_df.head(5).iterrows():
        print('  ' + row['feature'] + ': ' + str(round(row['mean_abs_shap'], 4)))
        
    shap_summary_path = os.path.join(data_dir, 'shap_summary_step5.csv')
    shap_df.to_csv(shap_summary_path, index=False)
    print('\nSHAP summary saved to ' + shap_summary_path)
    
    plt.rcParams['text.usetex'] = False
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values_pos, X_full_df, plot_type='bar', show=False)
    plt.title('SHAP Summary Bar Plot', fontsize=14)
    plt.tight_layout()
    
    timestamp = int(time.time())
    shap_plot_filename = 'shap_summary_bar_1_' + str(timestamp) + '.png'
    shap_plot_filepath = os.path.join(data_dir, shap_plot_filename)
    plt.savefig(shap_plot_filepath, dpi=300)
    plt.close()
    print('SHAP summary bar plot saved to ' + shap_plot_filepath)
    
    print('\nGenerating 2D Partial Dependence Plot (ALE proxy)...')
    fig, ax = plt.subplots(figsize=(8, 6))
    
    display = PartialDependenceDisplay.from_estimator(
        best_full_model, 
        X_full_df, 
        features=[('d_band_filling', 'en_difference')],
        grid_resolution=30,
        ax=ax
    )
    
    stable_mask = train_df['is_stable'] == True
    stable_d_band = train_df.loc[stable_mask, 'd_band_filling']
    stable_en_diff = train_df.loc[stable_mask, 'en_difference']
    
    ax.scatter(stable_d_band, stable_en_diff, color='red', edgecolor='white', 
               s=60, label='Stable Materials', zorder=10)
               
    ax.set_title('2D PDP: d_band_filling vs en_difference', fontsize=14)
    ax.legend(loc='best')
    plt.tight_layout()
    
    ale_plot_filename = 'ale_2d_plot_1_' + str(timestamp) + '.png'
    ale_plot_filepath = os.path.join(data_dir, ale_plot_filename)
    plt.savefig(ale_plot_filepath, dpi=300)
    plt.close()
    print('2D ALE/PDP plot saved to ' + ale_plot_filepath)