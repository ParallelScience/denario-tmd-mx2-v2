# filename: codebase/step_2.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np
import os
import json

def main():
    data_dir = 'data'
    data_path = os.path.join(data_dir, 'processed_tmd_data.csv')
    df = pd.read_csv(data_path)
    elastic_df = df[df['G_vrh'].notna()].copy()
    elastic_df = elastic_df.reset_index(drop=True)
    median_G_vrh = elastic_df['G_vrh'].median()
    elastic_df['is_robust'] = (elastic_df['G_vrh'] > median_G_vrh).astype(int)
    print('--- Global Target Definition ---')
    print('Median G_vrh threshold: ' + str(round(median_G_vrh, 4)) + ' GPa')
    print('Total samples in elastic subset: ' + str(len(elastic_df)))
    print('Robust samples (is_robust=1): ' + str(elastic_df['is_robust'].sum()))
    print('Non-robust samples (is_robust=0): ' + str(len(elastic_df) - elastic_df['is_robust'].sum()))
    print('--------------------------------\n')
    metals = elastic_df['metal'].unique()
    lomo_cv = {}
    print('--- LOMO CV Splits ---')
    for m in sorted(metals):
        test_idx = elastic_df[elastic_df['metal'] == m].index.tolist()
        train_idx = elastic_df[elastic_df['metal'] != m].index.tolist()
        lomo_cv[m] = {'train': train_idx, 'test': test_idx}
        print('Metal ' + m + ': Train=' + str(len(train_idx)) + ', Test=' + str(len(test_idx)))
    print('----------------------\n')
    elastic_output_path = os.path.join(data_dir, 'elastic_subset.csv')
    elastic_df.to_csv(elastic_output_path, index=False)
    print('Elastic subset with target saved to ' + elastic_output_path)
    cv_output_path = os.path.join(data_dir, 'lomo_cv_indices.json')
    with open(cv_output_path, 'w') as f:
        json.dump(lomo_cv, f)
    print('LOMO CV indices saved to ' + cv_output_path)

if __name__ == '__main__':
    main()