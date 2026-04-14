# filename: codebase/step_5.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np
import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

plt.rcParams['text.usetex'] = False

if __name__ == '__main__':
    data_dir = 'data/'
    orig_data_path = '/home/node/work/projects/materials_project_v1/tmd_data_enriched.csv'
    df_orig = pd.read_csv(orig_data_path)
    df_proc_path = os.path.join(data_dir, 'processed_tmd_data_with_target.csv')
    df_proc = pd.read_csv(df_proc_path)
    results_path = os.path.join(data_dir, 'model_training_results.json')
    with open(results_path, 'r') as f:
        results = json.load(f)
    features = results['features']
    model_path = os.path.join(data_dir, 'rf_model.joblib')
    model = joblib.load(model_path)
    cand_mask = df_proc['is_robust'].isna()
    df_cand_proc = df_proc[cand_mask]
    df_cand_orig = df_orig[cand_mask].copy()
    print('Number of metastable candidates: ' + str(len(df_cand_orig)))
    X_cand = df_cand_proc[features]
    probs = model.predict_proba(X_cand)[:, 1]
    df_cand_orig['predicted_is_robust_probability'] = probs
    cols_to_keep = ['material_id', 'formula', 'metal', 'chalcogen', 'phase', 'energy_above_hull', 'predicted_is_robust_probability']
    df_results = df_cand_orig[cols_to_keep].copy()
    df_results = df_results.sort_values(by='predicted_is_robust_probability', ascending=False).reset_index(drop=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_csv_path = os.path.join(data_dir, 'metastable_candidates_predictions_' + timestamp + '.csv')
    df_results.to_csv(results_csv_path, index=False)
    print('Ranked probability table saved to ' + results_csv_path)
    top_candidates = df_results[(df_results['predicted_is_robust_probability'] > 0.8) & (df_results['energy_above_hull'] < 0.05)]
    print('\nCandidates passing the dual filter (Prob > 0.8 and E_hull < 0.05 eV/atom): ' + str(len(top_candidates)))
    if len(top_candidates) > 0:
        print(top_candidates.to_string(index=False))
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.scatterplot(data=df_cand_orig, x='energy_above_hull', y='predicted_is_robust_probability', hue='phase', palette='tab10', s=80, alpha=0.8, ax=ax)
    ax.axhline(y=0.8, color='red', linestyle='--', label='Probability = 0.8')
    ax.axvline(x=0.05, color='blue', linestyle='--', label='E_hull = 0.05 eV/atom')
    ax.fill_betweenx([0.8, 1.05], -0.05, 0.05, color='green', alpha=0.1, label='Promising Region')
    ax.set_xlabel('Energy Above Hull (eV/atom)')
    ax.set_ylabel('Predicted Probability of being Robust')
    ax.set_title('Mechanical Robustness vs Thermodynamic Stability for Metastable TMDs')
    ax.set_xlim(left=-0.02)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    fig.tight_layout()
    plot_path = os.path.join(data_dir, 'robustness_vs_stability_scatter_1_' + timestamp + '.png')
    fig.savefig(plot_path, dpi=300)
    print('\nScatter plot saved to ' + plot_path)
    plt.close(fig)