# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np
import os

if __name__ == '__main__':
    data_path = "/home/node/work/projects/materials_project_v1/tmd_data_enriched.csv"
    df = pd.read_csv(data_path)
    print("Original dataset shape: " + str(df.shape))
    df['pugh_ratio'] = df['G_vrh'] / df['K_vrh']
    df['is_dos_missing'] = df['dos_at_fermi'].isna().astype(int)
    df['is_high_energy'] = (df['energy_above_hull'] > 0.05).astype(int)
    df = pd.get_dummies(df, columns=['crystal_system', 'phase', 'magnetic_ordering'], dummy_na=False)
    output_path = os.path.join("data", "processed_tmd_data.csv")
    df.to_csv(output_path, index=False)
    print("Number of samples with elastic data (Pugh's ratio calculated): " + str(df['pugh_ratio'].notna().sum()))
    print("Number of samples with missing dos_at_fermi: " + str(df['is_dos_missing'].sum()))
    print("Number of high energy samples (> 0.05 eV/atom): " + str(df['is_high_energy'].sum()))
    print("Processed dataset saved to: " + output_path)
    print("Processed dataset shape: " + str(df.shape))
    print("\nSummary of Pugh's ratio (dimensionless):")
    print(df['pugh_ratio'].describe().to_string())
    print("\nList of one-hot encoded columns added:")
    encoded_cols = [col for col in df.columns if col.startswith('crystal_system_') or col.startswith('phase_') or col.startswith('magnetic_ordering_')]
    for col in encoded_cols:
        print(" - " + col)