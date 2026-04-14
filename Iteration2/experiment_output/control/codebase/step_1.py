# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import NearestNeighbors
from scipy.stats import ks_2samp
import os

if __name__ == '__main__':
    data_path = '/home/node/work/projects/materials_project_v1/tmd_data_enriched.csv'
    df = pd.read_csv(data_path)
    df['is_dos_missing'] = df['dos_at_fermi'].isna().astype(int)
    has_elastic = df['G_vrh'].notna()
    missing_elastic = df['G_vrh'].isna()
    ks_d_band = ks_2samp(df['d_band_filling'][has_elastic], df['d_band_filling'][missing_elastic])
    ks_M_en = ks_2samp(df['M_en'][has_elastic], df['M_en'][missing_elastic])
    categorical_cols = ['metal', 'chalcogen', 'crystal_system', 'phase', 'magnetic_ordering', 'spacegroup_symbol']
    target_col = 'is_stable'
    for col in categorical_cols:
        if col in df.columns:
            global_mean = df[target_col].mean()
            agg = df.groupby(col)[target_col].agg(['count', 'mean'])
            weight = 10.0
            smooth = (agg['count'] * agg['mean'] + weight * global_mean) / (agg['count'] + weight)
            df[col + '_encoded'] = df[col].map(smooth)
            df[col + '_encoded'] = df[col + '_encoded'].fillna(global_mean)
    features_for_knn_base = ['volume', 'nsites', 'volume_per_atom', 'a', 'b', 'c', 'alpha', 'beta', 'gamma', 'c_a_ratio', 'energy_above_hull', 'log1p_energy_above_hull', 'formation_energy_per_atom', 'band_gap', 'efermi', 'total_magnetization', 'M_val', 'M_Z', 'M_en', 'M_ie1', 'M_atomic_radius', 'M_group', 'M_period', 'M_soc_proxy', 'X_en', 'X_ie1', 'X_atomic_radius', 'X_period', 'en_difference', 'bond_radius_sum', 'd_count_m4plus', 'd_band_filling']
    features_for_knn_base = [c for c in features_for_knn_base if c in df.columns]
    scaler_knn = RobustScaler()
    scaled_knn_features = scaler_knn.fit_transform(df[features_for_knn_base])
    df_knn_features = pd.DataFrame(scaled_knn_features, columns=features_for_knn_base, index=df.index)
    for col in categorical_cols:
        encoded_col = col + '_encoded'
        if encoded_col in df.columns:
            df_knn_features[encoded_col] = df[encoded_col]
    missing_mask = df['is_dos_missing'] == 1
    df_missing_knn = df_knn_features[missing_mask]
    df_valid_knn = df_knn_features[~missing_mask]
    if missing_mask.sum() > 0:
        nn = NearestNeighbors(n_neighbors=5)
        nn.fit(df_valid_knn)
        distances, indices = nn.kneighbors(df_missing_knn)
        valid_dos_orig = df.loc[~missing_mask, 'dos_at_fermi'].values
        neighbor_dos_orig = valid_dos_orig[indices]
        imputed_means_orig = np.mean(neighbor_dos_orig, axis=1)
        imputed_stds_orig = np.std(neighbor_dos_orig, axis=1, ddof=1)
        df.loc[missing_mask, 'dos_at_fermi'] = imputed_means_orig
        df['imputation_uncertainty'] = 0.0
        df.loc[missing_mask, 'imputation_uncertainty'] = imputed_stds_orig
    else:
        df['imputation_uncertainty'] = 0.0
        imputed_stds_orig = []
    continuous_cols = features_for_knn_base + ['dos_at_fermi', 'imputation_uncertainty']
    scaler = RobustScaler()
    df[continuous_cols] = scaler.fit_transform(df[continuous_cols])
    print('--- KNN Imputation Summary ---')
    print('Number of missing dos_at_fermi values imputed: ' + str(missing_mask.sum()))
    if missing_mask.sum() > 0:
        print('Range of imputation_uncertainty for imputed values (original units): [' + str(np.min(imputed_stds_orig)) + ', ' + str(np.max(imputed_stds_orig)) + ']')
    else:
        print('Range of imputation_uncertainty: [0.0, 0.0]')
    print('\n--- Kolmogorov-Smirnov Test for Dataset Shift ---')
    print('Feature: d_band_filling')
    print('  KS Statistic: ' + str(ks_d_band.statistic))
    print('  p-value: ' + str(ks_d_band.pvalue))
    print('Feature: M_en')
    print('  KS Statistic: ' + str(ks_M_en.statistic))
    print('  p-value: ' + str(ks_M_en.pvalue))
    output_path = 'data/tmd_data_preprocessing_step1.csv'
    df.to_csv(output_path, index=False)
    print('\nProcessed dataframe saved to ' + output_path)