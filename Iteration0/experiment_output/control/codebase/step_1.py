# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
import os

if __name__ == '__main__':
    file_path = '/home/node/work/projects/materials_project_v1/tmd_data_enriched.csv'
    df = pd.read_csv(file_path)
    print('Original dataset shape: ' + str(df.shape))
    df['phase_raw'] = df['phase']
    df = pd.get_dummies(df, columns=['magnetic_ordering', 'phase'], dummy_na=False, dtype=int)
    df.rename(columns={'phase_raw': 'phase'}, inplace=True)
    symmetry_order = ['Triclinic', 'Monoclinic', 'Orthorhombic', 'Tetragonal', 'Trigonal', 'Hexagonal', 'Cubic']
    present_systems = df['crystal_system'].dropna().unique().tolist()
    ordered_categories = [cs for cs in symmetry_order if cs in present_systems] + [cs for cs in present_systems if cs not in symmetry_order]
    encoder = OrdinalEncoder(categories=[ordered_categories])
    df['crystal_system_encoded'] = encoder.fit_transform(df[['crystal_system']])
    df.drop(columns=['crystal_system'], inplace=True)
    df.rename(columns={'crystal_system_encoded': 'crystal_system'}, inplace=True)
    continuous_features = ['volume', 'nsites', 'volume_per_atom', 'a', 'b', 'c', 'alpha', 'beta', 'gamma', 'c_a_ratio', 'energy_above_hull', 'log1p_energy_above_hull', 'formation_energy_per_atom', 'band_gap', 'efermi', 'dos_at_fermi', 'total_magnetization', 'M_val', 'M_Z', 'M_en', 'M_ie1', 'M_atomic_radius', 'M_group', 'M_period', 'M_soc_proxy', 'X_en', 'X_ie1', 'X_atomic_radius', 'X_period', 'en_difference', 'bond_radius_sum', 'd_count_m4plus', 'd_band_filling']
    scaler = StandardScaler()
    df[continuous_features] = scaler.fit_transform(df[continuous_features])
    output_path = os.path.join('data', 'processed_tmd_data.csv')
    df.to_csv(output_path, index=False)
    print('Processed dataset shape: ' + str(df.shape))
    print('Processed data saved to: ' + output_path)
    print('\nSummary of standardized continuous features (mean and std):')
    for col in ['c_a_ratio', 'volume_per_atom', 'energy_above_hull', 'd_band_filling', 'en_difference', 'M_soc_proxy']:
        print(col + ' - Mean: ' + str(round(df[col].mean(), 4)) + ', Std: ' + str(round(df[col].std(), 4)))