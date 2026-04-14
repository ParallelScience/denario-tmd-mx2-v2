# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np
import json
import os
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def main():
    data_path = '/home/node/work/projects/materials_project_v1/tmd_data_enriched.csv'
    df = pd.read_csv(data_path)
    print('=== Dataset Statistics ===')
    print('Total materials: ' + str(len(df)))
    print('\nMissing values in original dataset:')
    missing = df.isnull().sum()
    print(missing[missing > 0].to_string())
    categorical_cols = ['magnetic_ordering', 'crystal_system', 'phase']
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=False)
    for col in df_encoded.columns:
        if df_encoded[col].dtype == bool:
            df_encoded[col] = df_encoded[col].astype(int)
    known_df = df_encoded[df_encoded['G_vrh'].notna()].copy()
    unknown_df = df_encoded[df_encoded['G_vrh'].isna()].copy()
    print('\nElasticity-Known subset: ' + str(len(known_df)) + ' samples')
    print('Elasticity-Unknown subset: ' + str(len(unknown_df)) + ' samples')
    print('\nG_vrh distribution in Elasticity-Known subset:')
    print(known_df['G_vrh'].describe().to_string())
    known_df['dos_at_fermi_raw'] = known_df['dos_at_fermi']
    unknown_df['dos_at_fermi_raw'] = unknown_df['dos_at_fermi']
    exclude_from_imputation = ['G_vrh', 'K_vrh', 'poisson_ratio', 'elastic_anisotropy', 'debye_temperature', 'is_stable', 'energy_above_hull', 'log1p_energy_above_hull', 'material_id', 'formula', 'metal', 'chalcogen', 'Tc_phase', 'Tc_ref', 'dos_at_fermi_raw']
    numeric_cols = known_df.select_dtypes(include=[np.number]).columns.tolist()
    impute_features = [c for c in numeric_cols if c not in exclude_from_imputation]
    print('\nPerforming Iterative Imputation for dos_at_fermi (baseline fit on known_df)...')
    imputer = IterativeImputer(estimator=BayesianRidge(), random_state=42)
    imputer.fit(known_df[impute_features])
    known_df_imputed = known_df.copy()
    unknown_df_imputed = unknown_df.copy()
    known_df_imputed[impute_features] = imputer.transform(known_df[impute_features])
    unknown_df_imputed[impute_features] = imputer.transform(unknown_df[impute_features])
    continuous_cols = ['d_band_filling', 'en_difference', 'M_soc_proxy', 'c_a_ratio', 'bond_radius_sum', 'dos_at_fermi', 'volume_per_atom', 'formation_energy_per_atom', 'band_gap', 'efermi', 'total_magnetization']
    print('Standardizing continuous features...')
    scaler = StandardScaler()
    known_df_imputed[continuous_cols] = scaler.fit_transform(known_df_imputed[continuous_cols])
    unknown_df_imputed[continuous_cols] = scaler.transform(unknown_df_imputed[continuous_cols])
    print('Z-score normalizing G_vrh to [0, 1] range...')
    z_scaler = StandardScaler()
    g_vrh_z = z_scaler.fit_transform(known_df_imputed[['G_vrh']])
    minmax_scaler = MinMaxScaler()
    known_df_imputed['G_vrh_norm'] = minmax_scaler.fit_transform(g_vrh_z)
    unknown_df_imputed['G_vrh_norm'] = np.nan
    one_hot_cols = [c for c in df_encoded.columns if c.startswith('magnetic_ordering_') or c.startswith('crystal_system_') or c.startswith('phase_')]
    feature_list = continuous_cols + one_hot_cols
    known_df_imputed.to_csv(os.path.join('data', 'known_df_processed.csv'), index=False)
    unknown_df_imputed.to_csv(os.path.join('data', 'unknown_df_processed.csv'), index=False)
    with open(os.path.join('data', 'feature_list.json'), 'w') as f:
        json.dump(feature_list, f)
    print('\nProcessed datasets saved to data/known_df_processed.csv and data/unknown_df_processed.csv')
    print('Feature list saved to data/feature_list.json')
    print('Number of features selected for modeling: ' + str(len(feature_list)))

if __name__ == '__main__':
    main()