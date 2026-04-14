# filename: codebase/step_6.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np
import joblib
import os

if __name__ == '__main__':
    data_dir = 'data/'
    test_path = os.path.join(data_dir, 'test_dataset_step2.csv')
    test_df = pd.read_csv(test_path)
    dist_path = os.path.join(data_dir, 'mahalanobis_distances_step4.csv')
    dist_df = pd.read_csv(dist_path)
    test_df = test_df.merge(dist_df[['material_id', 'mahalanobis_distance', 'is_ood']], on='material_id', how='left')
    clf_path = os.path.join(data_dir, 'full_rf_classifier.joblib')
    reg_path = os.path.join(data_dir, 'full_rf_regressor.joblib')
    clf = joblib.load(clf_path)
    reg = joblib.load(reg_path)
    full_features = ['X_atomic_radius', 'crystal_system_encoded', 'M_group', 'd_band_filling', 'dos_at_fermi', 'M_soc_proxy', 'en_difference', 'magnetic_ordering_encoded', 'phase_encoded', 'imputation_uncertainty', 'is_dos_missing']
    X_test = test_df[full_features]
    test_df['prob_is_robust'] = clf.predict_proba(X_test)[:, 1]
    test_df['pred_elastic_anisotropy'] = reg.predict(X_test)
    ranked_df = test_df.sort_values(by=['prob_is_robust', 'pred_elastic_anisotropy'], ascending=[False, True])
    anisotropy_90th = ranked_df['pred_elastic_anisotropy'].quantile(0.90)
    filtered_df = ranked_df[(ranked_df['prob_is_robust'] > 0.75) & (ranked_df['pred_elastic_anisotropy'] < anisotropy_90th) & (~ranked_df['is_ood']) & (ranked_df['imputation_uncertainty'] <= 1.0)]
    cols_to_print = ['material_id', 'formula', 'metal', 'chalcogen', 'phase', 'prob_is_robust', 'pred_elastic_anisotropy', 'imputation_uncertainty', 'mahalanobis_distance', 'is_ood']
    print('--- Full Prioritized Candidate List ---')
    print('Total candidates before filtering: ' + str(len(ranked_df)))
    print('Total candidates after filtering: ' + str(len(filtered_df)))
    print('Filters applied: prob_is_robust > 0.75, pred_elastic_anisotropy < ' + str(round(anisotropy_90th, 2)) + ', is_ood == False, imputation_uncertainty <= 1.0\n')
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    if len(filtered_df) > 0:
        final_df = filtered_df[cols_to_print].copy()
    else:
        print('No candidates met all the filtering criteria. Showing top 15 candidates without strict filtering:')
        final_df = ranked_df.head(15)[cols_to_print].copy()
    display_df = final_df.copy()
    display_df['prob_is_robust'] = display_df['prob_is_robust'].round(4)
    display_df['pred_elastic_anisotropy'] = display_df['pred_elastic_anisotropy'].round(4)
    display_df['imputation_uncertainty'] = display_df['imputation_uncertainty'].round(4)
    display_df['mahalanobis_distance'] = display_df['mahalanobis_distance'].round(4)
    print(display_df.to_string(index=False))
    output_path = os.path.join(data_dir, 'prioritized_candidates_step6.csv')
    final_df.to_csv(output_path, index=False)
    print('\nPrioritized candidate list saved to ' + output_path)