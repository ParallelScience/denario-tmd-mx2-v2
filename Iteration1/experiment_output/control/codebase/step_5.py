# filename: codebase/step_5.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

def main():
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    data_dir = 'data'
    
    df = pd.read_csv(os.path.join(data_dir, 'processed_tmd_data.csv'))
    
    df_train = df[df['G_vrh'].notna()].copy().reset_index(drop=True)
    df_cand = df[df['G_vrh'].isna()].copy().reset_index(drop=True)
    
    with open(os.path.join(data_dir, 'lomo_cv_results.json'), 'r') as f:
        lomo_results = json.load(f)
        
    hp_counts = {}
    for metal, res in lomo_results.items():
        hp = res['best_hp']
        hp_str = 'n_est=' + str(hp['n_estimators']) + ', max_depth=' + str(hp['max_depth']) + ', min_samples_split=' + str(hp['min_samples_split'])
        if hp_str not in hp_counts:
            hp_counts[hp_str] = {'count': 0, 'hp': hp, 'inner_auprc_sum': 0}
        hp_counts[hp_str]['count'] += 1
        hp_counts[hp_str]['inner_auprc_sum'] += res['inner_auprc']
        
    best_hp_str = max(hp_counts.keys(), key=lambda k: (hp_counts[k]['count'], hp_counts[k]['inner_auprc_sum']))
    best_hp = hp_counts[best_hp_str]['hp']
    
    print('--- Final Model Hyperparameters ---')
    print('Selected most frequent best HP from CV: ' + best_hp_str)
    print('Frequency: ' + str(hp_counts[best_hp_str]['count']) + ' out of ' + str(len(lomo_results)) + ' folds')
    print('-----------------------------------\n')
    
    knn_features = ['band_gap', 'd_band_filling', 'en_difference', 'M_soc_proxy']
    valid_train = df_train[~df_train['dos_at_fermi_imputed_flag']]
    
    if len(valid_train) > 0:
        knn = KNeighborsRegressor(n_neighbors=min(5, len(valid_train)))
        knn.fit(valid_train[knn_features], valid_train['dos_at_fermi'])
        
        impute_train_mask = df_train['dos_at_fermi_imputed_flag']
        if impute_train_mask.sum() > 0:
            df_train.loc[impute_train_mask, 'dos_at_fermi'] = knn.predict(df_train.loc[impute_train_mask, knn_features])
            
        impute_cand_mask = df_cand['dos_at_fermi_imputed_flag']
        if impute_cand_mask.sum() > 0:
            df_cand.loc[impute_cand_mask, 'dos_at_fermi'] = knn.predict(df_cand.loc[impute_cand_mask, knn_features])
    else:
        df_train.loc[df_train['dos_at_fermi_imputed_flag'], 'dos_at_fermi'] = 0.0
        df_cand.loc[df_cand['dos_at_fermi_imputed_flag'], 'dos_at_fermi'] = 0.0
        
    median_G_vrh = df_train['G_vrh'].median()
    df_train['is_robust'] = (df_train['G_vrh'] > median_G_vrh).astype(int)
    
    print('--- Training Final Model ---')
    print('Median G_vrh threshold used for training: ' + str(round(median_G_vrh, 4)) + ' GPa')
    print('Training samples: ' + str(len(df_train)))
    print('Robust samples (class 1): ' + str(df_train['is_robust'].sum()))
    print('----------------------------\n')
    
    ohe_features = [c for c in df.columns if c.startswith('magnetic_ordering_') or c.startswith('phase_')]
    rf_features = ['d_band_filling', 'M_soc_proxy', 'en_difference', 'M_atomic_radius', 'X_atomic_radius', 'dos_at_fermi', 'crystal_system_ord'] + ohe_features
    
    rf_final = RandomForestClassifier(
        n_estimators=best_hp['n_estimators'],
        max_depth=best_hp['max_depth'],
        min_samples_split=best_hp['min_samples_split'],
        class_weight='balanced',
        random_state=42,
        n_jobs=1
    )
    rf_final.fit(df_train[rf_features], df_train['is_robust'])
    
    df_cand['prob_robust'] = rf_final.predict_proba(df_cand[rf_features])[:, 1]
    
    aniso_threshold = df_train['elastic_anisotropy'].quantile(0.90)
    print('--- Elastic Anisotropy Filter ---')
    print('90th-percentile threshold from training set: ' + str(round(aniso_threshold, 4)))
    
    train_aniso = df_train.dropna(subset=['elastic_anisotropy'])
    rf_aniso = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1)
    rf_aniso.fit(train_aniso[rf_features], train_aniso['elastic_anisotropy'])
    
    df_cand['pred_elastic_anisotropy'] = rf_aniso.predict(df_cand[rf_features])
    df_cand['final_elastic_anisotropy'] = df_cand['elastic_anisotropy'].fillna(df_cand['pred_elastic_anisotropy'])
    
    print('--- Predicting G_vrh for Candidates ---')
    rf_gvrh = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1)
    rf_gvrh.fit(df_train[rf_features], df_train['G_vrh'])
    df_cand['pred_G_vrh'] = rf_gvrh.predict(df_cand[rf_features])
    print('G_vrh predicted for all candidates using RandomForestRegressor.')
    print('-----------------------------------------\n')
    
    mask_aniso = df_cand['final_elastic_anisotropy'] <= aniso_threshold
    df_cand_filtered = df_cand[mask_aniso].copy()
    
    print('Total metastable candidates: ' + str(len(df_cand)))
    print('Candidates passing anisotropy filter (<= ' + str(round(aniso_threshold, 4)) + '): ' + str(len(df_cand_filtered)))
    print('Candidates rejected by filter: ' + str(len(df_cand) - len(df_cand_filtered)))
    print('---------------------------------\n')
    
    df_cand_filtered = df_cand_filtered.sort_values(by='prob_robust', ascending=False)
    
    high_priority = df_cand_filtered[df_cand_filtered['prob_robust'] > 0.75]
    
    print('--- High-Priority Candidates (prob > 0.75 & acceptable anisotropy) ---')
    print('Total high-priority candidates found: ' + str(len(high_priority)))
    if len(high_priority) > 0:
        cols_to_print = ['material_id', 'formula', 'prob_robust', 'final_elastic_anisotropy', 'pred_G_vrh']
        print(high_priority[cols_to_print].to_string(index=False))
    print('----------------------------------------------------------------------\n')
    
    print('--- Top 10 Candidates by Robustness Probability ---')
    cols_to_print = ['material_id', 'formula', 'prob_robust', 'final_elastic_anisotropy', 'pred_G_vrh']
    print(df_cand_filtered.head(10)[cols_to_print].to_string(index=False))
    print('---------------------------------------------------\n')
    
    output_path = os.path.join(data_dir, 'prioritized_candidates.csv')
    df_cand_filtered.to_csv(output_path, index=False)
    print('Filtered and ranked candidates saved to ' + output_path)
    
    full_output_path = os.path.join(data_dir, 'all_candidates_predictions.csv')
    df_cand.to_csv(full_output_path, index=False)
    print('All candidates with predictions saved to ' + full_output_path)

if __name__ == '__main__':
    main()