# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np
import os

def process_data():
    file_path = "/home/node/work/projects/materials_project_v1/tmd_data_enriched.csv"
    df = pd.read_csv(file_path)
    df['pugh_ratio'] = df['G_vrh'] / df['K_vrh']
    df['is_dos_missing'] = df['dos_at_fermi'].isna().astype(int)
    df = pd.get_dummies(df, columns=['crystal_system', 'phase'], dummy_na=False, drop_first=False, dtype=int)
    df = pd.get_dummies(df, columns=['magnetic_ordering'], dummy_na=False, drop_first=False, dtype=int)
    df['high_energy_above_hull'] = (df['energy_above_hull'] > 0.05).astype(int)
    if 'c_a_ratio' not in df.columns:
        df['c_a_ratio'] = df['c'] / df['a']
    output_path = os.path.join("data", "processed_tmd_data.csv")
    df.to_csv(output_path, index=False)
    print("Dataset loaded from: " + file_path)
    print("Original shape: " + str(pd.read_csv(file_path).shape))
    print("Processed shape: " + str(df.shape))
    print("Number of samples with Pugh's ratio: " + str(df['pugh_ratio'].notna().sum()))
    print("Number of samples with missing dos_at_fermi: " + str(df['is_dos_missing'].sum()))
    print("Number of samples with energy_above_hull > 0.05: " + str(df['high_energy_above_hull'].sum()))
    print("Processed dataset saved to: " + output_path)

if __name__ == '__main__':
    process_data()