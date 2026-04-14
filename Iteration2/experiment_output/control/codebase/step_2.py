# filename: codebase/step_2.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np
import os

if __name__ == '__main__':
    data_dir = 'data/'
    data_path = os.path.join(data_dir, 'tmd_data_preprocessing_step1.csv')
    df = pd.read_csv(data_path)
    
    elastic_mask = df['G_vrh'].notna()
    
    stable_elastic = df[(df['is_stable'] == True) & elastic_mask]
    threshold = stable_elastic['G_vrh'].quantile(0.70)
    
    df['is_robust'] = np.nan
    df.loc[elastic_mask, 'is_robust'] = (df.loc[elastic_mask, 'G_vrh'] >= threshold).astype(float)
    
    print('Calculated G_vrh threshold (70th percentile of stable subset): ' + str(round(threshold, 2)) + ' GPa')
    
    robust_counts = df.loc[elastic_mask, 'is_robust'].value_counts()
    robust_fractions = df.loc[elastic_mask, 'is_robust'].value_counts(normalize=True)
    
    count_true = robust_counts.get(1.0, 0)
    count_false = robust_counts.get(0.0, 0)
    frac_true = robust_fractions.get(1.0, 0.0)
    frac_false = robust_fractions.get(0.0, 0.0)
    
    print('Class balance of is_robust (90-sample elastic subset):')
    print('  True (1.0): ' + str(count_true) + ' (' + str(round(frac_true * 100, 2)) + '%)')
    print('  False (0.0): ' + str(count_false) + ' (' + str(round(frac_false * 100, 2)) + '%)')
    
    df['is_stable_structure'] = df['elastic_anisotropy']
    
    df.loc[~elastic_mask, 'is_stable_structure'] = np.nan
    df.loc[~elastic_mask, 'is_robust'] = np.nan
    
    train_df = df[elastic_mask].copy()
    test_df = df[~elastic_mask].copy()
    
    train_path = os.path.join(data_dir, 'train_dataset_step2.csv')
    test_path = os.path.join(data_dir, 'test_dataset_step2.csv')
    full_path = os.path.join(data_dir, 'full_dataset_step2.csv')
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    df.to_csv(full_path, index=False)
    
    print('\nPrepared datasets saved to:')
    print('  Training set (90 samples): ' + train_path)
    print('  Test set (112 samples): ' + test_path)
    print('  Full dataset (202 samples): ' + full_path)