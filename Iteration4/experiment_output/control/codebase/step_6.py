# filename: codebase/step_6.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE, RandomOverSampler

if __name__ == '__main__':
    data_dir = 'data/'
    candidates_df = pd.read_csv(os.path.join(data_dir, 'metastable_candidates_predictions.csv'))
    if 'is_exotic' in candidates_df.columns:
        initial_count = len(candidates_df)
        candidates_df = candidates_df[~candidates_df['is_exotic']]
        print('Excluded ' + str(initial_count - len(candidates_df)) + ' chemically exotic candidates.')
    candidates_df = candidates_df.sort_values(by='predicted_viability_probability', ascending=False)
    filtered_candidates = candidates_df[(candidates_df['predicted_viability_probability'] > 0.70) & (candidates_df['energy_above_hull'] < 0.05)].copy()
    print('Number of candidates after initial filter (prob > 0.70, energy < 0.05): ' + str(len(filtered_candidates)))
    if len(filtered_candidates) == 0:
        print('No candidates passed the initial filter.')
        sys.exit(0)
    train_df = pd.read_csv(os.path.join(data_dir, 'training_dataset_step2.csv'))
    with open(os.path.join(data_dir, 'selected_features.txt'), 'r') as f:
        selected_features = [line.strip() for line in f.readlines() if line.strip()]
    unique_metals = train_df['metal'].unique()
    print('Metals in training set for leave-one-metal-out: ' + ', '.join(unique_metals))
    X_cand = filtered_candidates[selected_features]
    sensitivity_probs = {metal: [] for metal in unique_metals}
    for metal in unique_metals:
        train_subset = train_df[train_df['metal'] != metal]
        X_train = train_subset[selected_features]
        y_train = train_subset['is_viable'].astype(int)
        if len(np.unique(y_train)) > 1:
            min_class_count = y_train.value_counts().min()
            k_neighbors = min(5, min_class_count - 1)
            if k_neighbors > 0:
                smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
                X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
            else:
                ros = RandomOverSampler(random_state=42)
                X_train_res, y_train_res = ros.fit_resample(X_train, y_train)
        else:
            X_train_res, y_train_res = X_train, y_train
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train_res, y_train_res)
        if len(np.unique(y_train_res)) > 1:
            probs = rf.predict_proba(X_cand)[:, 1]
        else:
            single_class = y_train_res.iloc[0] if isinstance(y_train_res, pd.Series) else y_train_res[0]
            if single_class == 1:
                probs = np.ones(len(X_cand))
            else:
                probs = np.zeros(len(X_cand))
        sensitivity_probs[metal] = probs
    min_probs = np.zeros(len(filtered_candidates))
    for i in range(len(filtered_candidates)):
        min_prob = min([sensitivity_probs[metal][i] for metal in unique_metals])
        min_probs[i] = min_prob
    filtered_candidates['min_sensitivity_prob'] = min_probs
    filtered_candidates['is_sensitivity_dependent'] = filtered_candidates['min_sensitivity_prob'] < 0.70
    final_candidates = filtered_candidates[~filtered_candidates['is_sensitivity_dependent']]
    print('\nSensitivity Analysis Results:')
    print('  - Candidates flagged as sensitivity-dependent (prob dropped < 0.70): ' + str(filtered_candidates['is_sensitivity_dependent'].sum()))
    print('  - Final robust candidates: ' + str(len(final_candidates)))
    print('\nFinal Prioritized List:')
    cols_to_print = ['material_id', 'formula', 'phase', 'magnetic_ordering', 'energy_above_hull', 'predicted_viability_probability']
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    if len(final_candidates) > 0:
        print(final_candidates[cols_to_print].to_string(index=False))
    else:
        print('No candidates remained after sensitivity analysis.')
    final_candidates.to_csv(os.path.join(data_dir, 'final_prioritized_candidates.csv'), index=False)