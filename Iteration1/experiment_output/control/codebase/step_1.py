# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.neighbors import KNeighborsRegressor
import os

def main():
    file_path = "/home/node/work/projects/materials_project_v1/tmd_data_enriched.csv"
    df = pd.read_csv(file_path)
    missing_dos = df['dos_at_fermi'].isna()
    semi_mask = missing_dos & (df['band_gap'] > 0.1)
    df.loc[semi_mask, 'dos_at_fermi'] = 0.0
    count_zero_set = semi_mask.sum()
    metal_mask = missing_dos & (df['band_gap'] <= 0.1)
    count_knn_imputed = metal_mask.sum()
    if count_knn_imputed > 0:
        knn_features = ['band_gap', 'd_band_filling', 'en_difference', 'M_soc_proxy']
        train_mask = ~df['dos_at_fermi'].isna()
        scaler_knn = StandardScaler()
        X_train = scaler_knn.fit_transform(df.loc[train_mask, knn_features])
        y_train = df.loc[train_mask, 'dos_at_fermi']
        X_test = scaler_knn.transform(df.loc[metal_mask, knn_features])
        knn = KNeighborsRegressor(n_neighbors=5)
        knn.fit(X_train, y_train)
        imputed_values = knn.predict(X_test)
        df.loc[metal_mask, 'dos_at_fermi'] = imputed_values
    else:
        imputed_values = np.array([])
    print("--- Imputation Summary for dos_at_fermi ---")
    print("Count of zero-set values (semiconductors): " + str(count_zero_set))
    print("Count of KNN-imputed values (metals): " + str(count_knn_imputed))
    if count_knn_imputed > 0:
        print("Distribution of KNN-imputed values (Global Fit):")
        print("  Mean: " + str(round(imputed_values.mean(), 4)))
        print("  Std:  " + str(round(imputed_values.std(), 4)))
        print("  Min:  " + str(round(imputed_values.min(), 4)))
        print("  Max:  " + str(round(imputed_values.max(), 4)))
    print("-------------------------------------------\n")
    df['dos_at_fermi_imputed_flag'] = metal_mask
    ohe_cols = ['magnetic_ordering', 'phase']
    try:
        ohe = OneHotEncoder(sparse_output=False, drop=None)
        ohe_data = ohe.fit_transform(df[ohe_cols])
    except TypeError:
        ohe = OneHotEncoder(sparse=False, drop=None)
        ohe_data = ohe.fit_transform(df[ohe_cols])
    ohe_df = pd.DataFrame(ohe_data, columns=ohe.get_feature_names_out(ohe_cols))
    oe = OrdinalEncoder()
    df['crystal_system_ord'] = oe.fit_transform(df[['crystal_system']])
    df = df.drop(columns=ohe_cols + ['crystal_system'])
    df = pd.concat([df, ohe_df], axis=1)
    continuous_features = ['volume', 'volume_per_atom', 'a', 'b', 'c', 'alpha', 'beta', 'gamma', 'c_a_ratio', 'energy_above_hull', 'log1p_energy_above_hull', 'formation_energy_per_atom', 'band_gap', 'efermi', 'dos_at_fermi', 'total_magnetization', 'M_en', 'M_ie1', 'M_atomic_radius', 'M_soc_proxy', 'X_en', 'X_ie1', 'X_atomic_radius', 'en_difference', 'bond_radius_sum', 'd_band_filling']
    scaler = StandardScaler()
    df[continuous_features] = scaler.fit_transform(df[continuous_features])
    output_path = os.path.join("data", "processed_tmd_data.csv")
    df.to_csv(output_path, index=False)
    print("Processed dataset saved to " + output_path)

if __name__ == '__main__':
    main()