# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.impute import KNNImputer
import os
from datetime import datetime

def main():
    plt.rcParams['text.usetex'] = False
    data_path = "/home/node/work/projects/materials_project_v1/tmd_data_enriched.csv"
    df = pd.read_csv(data_path)
    df['is_dos_missing'] = df['dos_at_fermi'].isna().astype(int)
    if 'is_known_SC' in df.columns:
        df['is_known_SC'] = df['is_known_SC'].fillna(False).astype(int)
    if 'is_stable' in df.columns:
        df['is_stable'] = df['is_stable'].astype(int)
    if 'theoretical' in df.columns:
        df['theoretical'] = df['theoretical'].astype(int)
    df = pd.get_dummies(df, columns=['phase', 'crystal_system', 'magnetic_ordering'], drop_first=False)
    one_hot_cols = [c for c in df.columns if c.startswith('phase_') or c.startswith('crystal_system_') or c.startswith('magnetic_ordering_')]
    for c in one_hot_cols:
        df[c] = df[c].astype(int)
    exclude_cols = ['material_id', 'formula', 'metal', 'chalcogen', 'spacegroup_symbol', 'Tc_phase', 'Tc_ref', 'is_dos_missing', 'is_stable', 'theoretical', 'is_known_SC'] + one_hot_cols
    continuous_features = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in exclude_cols]
    scaler = RobustScaler()
    df_scaled = df.copy()
    df_scaled[continuous_features] = scaler.fit_transform(df[continuous_features])
    impute_cols = continuous_features + one_hot_cols + ['is_stable', 'theoretical', 'is_known_SC']
    imputer = KNNImputer(n_neighbors=5)
    imputed_array = imputer.fit_transform(df_scaled[impute_cols])
    df_imputed_temp = pd.DataFrame(imputed_array, columns=impute_cols, index=df_scaled.index)
    df_imputed_original = df_imputed_temp.copy()
    df_imputed_original[continuous_features] = scaler.inverse_transform(df_imputed_temp[continuous_features])
    known_dos = df[df['is_dos_missing'] == 0]['dos_at_fermi']
    imputed_dos = df_imputed_original[df['is_dos_missing'] == 1]['dos_at_fermi']
    print("Summary statistics for KNOWN dos_at_fermi (states/eV):")
    print("Mean:   " + str(round(known_dos.mean(), 4)))
    print("Median: " + str(round(known_dos.median(), 4)))
    print("Std:    " + str(round(known_dos.std(), 4)))
    print("\nSummary statistics for IMPUTED dos_at_fermi (states/eV):")
    print("Mean:   " + str(round(imputed_dos.mean(), 4)))
    print("Median: " + str(round(imputed_dos.median(), 4)))
    print("Std:    " + str(round(imputed_dos.std(), 4)))
    plt.figure(figsize=(8, 6))
    sns.histplot(known_dos, label='Known', color='blue', kde=True, stat='density', alpha=0.4, bins=20)
    sns.histplot(imputed_dos, label='Imputed', color='orange', kde=True, stat='density', alpha=0.4, bins=10)
    plt.title('Distribution of Known vs Imputed dos_at_fermi')
    plt.xlabel('Density of States at Fermi Level (states/eV)')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = "dos_at_fermi_imputation_1_" + timestamp + ".png"
    plot_path = os.path.join("data", plot_filename)
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print("\nPlot saved to " + plot_path)
    df_scaled['dos_at_fermi'] = df_imputed_temp['dos_at_fermi']
    output_path = os.path.join("data", "processed_tmd_data.csv")
    df_scaled.to_csv(output_path, index=False)
    print("Processed dataframe saved to " + output_path)

if __name__ == '__main__':
    main()