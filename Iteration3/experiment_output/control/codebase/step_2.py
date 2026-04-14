# filename: codebase/step_2.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from datetime import datetime

def main():
    plt.rcParams['text.usetex'] = False
    data_path = "/home/node/work/projects/materials_project_v1/tmd_data_enriched.csv"
    orig_df = pd.read_csv(data_path)
    elastic_df = orig_df[orig_df['G_vrh'].notna()].copy()
    stable_elastic_df = elastic_df[elastic_df['is_stable'] == True]
    median_G_vrh = stable_elastic_df['G_vrh'].median()
    print("Median G_vrh of stable population: " + str(round(median_G_vrh, 4)) + " GPa\n")
    elastic_df['is_robust'] = (elastic_df['G_vrh'] > median_G_vrh).astype(int)
    print("Initial is_robust class counts (Threshold = Median):")
    print("Robust (1): " + str(elastic_df['is_robust'].sum()))
    print("Non-robust (0): " + str((elastic_df['is_robust'] == 0).sum()) + "\n")
    percentiles = [60, 65, 70, 75, 80]
    thresholds = []
    robust_counts = []
    non_robust_counts = []
    robust_sets = {}
    for p in percentiles:
        thresh = np.percentile(stable_elastic_df['G_vrh'], p)
        thresholds.append(thresh)
        is_robust_p = elastic_df['G_vrh'] > thresh
        robust_count = is_robust_p.sum()
        non_robust_count = (~is_robust_p).sum()
        robust_counts.append(robust_count)
        non_robust_counts.append(non_robust_count)
        robust_mids = elastic_df[is_robust_p]['material_id'].tolist()
        robust_sets[p] = set(robust_mids)
        elastic_df['is_robust_' + str(p)] = is_robust_p.astype(int)
        print("Percentile: " + str(p) + "th")
        print("Threshold: " + str(round(thresh, 4)) + " GPa")
        print("Class counts -> Robust: " + str(robust_count) + ", Non-robust: " + str(non_robust_count))
        print("High-confidence material IDs at this threshold: " + ", ".join(robust_mids) + "\n")
    high_confidence_set = robust_sets[60].intersection(robust_sets[65], robust_sets[70], robust_sets[75], robust_sets[80])
    print("Final High-confidence set (intersection across all tested thresholds):")
    print(", ".join(sorted(list(high_confidence_set))) + "\n")
    print("Size of high-confidence set: " + str(len(high_confidence_set)) + "\n")
    plt.figure(figsize=(8, 6))
    plt.plot(percentiles, robust_counts, marker='o', label='Robust (Class 1)', color='blue', linewidth=2)
    plt.plot(percentiles, non_robust_counts, marker='s', label='Non-robust (Class 0)', color='red', linewidth=2)
    plt.axhline(y=len(high_confidence_set), color='green', linestyle='--', label='Final High-confidence set size', linewidth=2)
    plt.title('Sensitivity of Class Balance to G_vrh Threshold')
    plt.xlabel('Percentile of Stable Population G_vrh')
    plt.ylabel('Number of Samples')
    plt.xticks(percentiles)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    timestamp = int(time.time())
    plot_filename = "threshold_sensitivity_2_" + str(timestamp) + ".png"
    plot_path = os.path.join("data", plot_filename)
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print("Saved to " + plot_path)
    target_cols = ['material_id', 'is_robust'] + ['is_robust_' + str(p) for p in percentiles]
    targets_df = elastic_df[target_cols]
    output_path = os.path.join("data", "target_labels.csv")
    targets_df.to_csv(output_path, index=False)
    print("Target labels saved to " + output_path)
if __name__ == '__main__':
    main()