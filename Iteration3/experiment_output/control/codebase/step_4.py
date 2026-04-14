# filename: codebase/step_4.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import LeaveOneGroupOut
import json
from scipy.stats import ttest_rel

def main():
    data_dir = "data"
    processed_data_path = os.path.join(data_dir, "processed_tmd_data.csv")
    targets_path = os.path.join(data_dir, "target_labels.csv")
    features_path = os.path.join(data_dir, "training_features.json")
    df_processed = pd.read_csv(processed_data_path)
    df_targets = pd.read_csv(targets_path)
    with open(features_path, "r") as f:
        base_features = json.load(f)
    df = pd.merge(df_targets[['material_id', 'is_robust']], df_processed, on='material_id', how='inner')
    y = df['is_robust']
    groups = df['metal']
    configs = {
        "Full Set": base_features,
        "No M_soc_proxy": [f for f in base_features if f != 'M_soc_proxy'],
        "No dos_at_fermi": [f for f in base_features if f != 'dos_at_fermi'],
        "No Both": [f for f in base_features if f not in ['M_soc_proxy', 'dos_at_fermi']]
    }
    logo = LeaveOneGroupOut()
    results = []
    fold_auprcs = {k: [] for k in configs.keys()}
    for config_name, features in configs.items():
        X = df[features]
        rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=8)
        auprcs = []
        for train_idx, test_idx in logo.split(X, y, groups):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            rf.fit(X_train, y_train)
            y_proba = rf.predict_proba(X_test)[:, 1]
            if len(np.unique(y_test)) > 1:
                auprc = average_precision_score(y_test, y_proba)
                auprcs.append(auprc)
            else:
                auprcs.append(np.nan)
        fold_auprcs[config_name] = auprcs
        mean_auprc = np.nanmean(auprcs)
        results.append({
            "Configuration": config_name,
            "Mean AUPRC": mean_auprc
        })
    print("Ablation Study Results (Macro-Avg AUPRC):")
    print("-" * 45)
    print("Configuration".ljust(20) + " | Mean AUPRC")
    print("-" * 45)
    for res in results:
        print(res["Configuration"].ljust(20) + " | " + str(round(res["Mean AUPRC"], 4)))
    print("-" * 45 + "\n")
    full_auprcs = np.array(fold_auprcs["Full Set"])
    no_soc_auprcs = np.array(fold_auprcs["No M_soc_proxy"])
    valid_idx = ~np.isnan(full_auprcs) & ~np.isnan(no_soc_auprcs)
    full_valid = full_auprcs[valid_idx]
    no_soc_valid = no_soc_auprcs[valid_idx]
    if len(full_valid) > 1:
        if np.all(full_valid == no_soc_valid):
            print("Paired t-test (Full Set vs No M_soc_proxy):")
            print("Mean AUPRC Difference: 0.0")
            print("p-value: 1.0")
            print("Conclusion: The performance drop from removing M_soc_proxy is statistically negligible (p > 0.05).")
        else:
            t_stat, p_val = ttest_rel(full_valid, no_soc_valid)
            mean_diff = np.mean(full_valid - no_soc_valid)
            print("Paired t-test (Full Set vs No M_soc_proxy):")
            print("Mean AUPRC Difference: " + str(round(mean_diff, 4)))
            print("p-value: " + str(round(p_val, 4)))
            if p_val > 0.05:
                print("Conclusion: The performance drop from removing M_soc_proxy is statistically negligible (p > 0.05).")
            else:
                print("Conclusion: The performance drop from removing M_soc_proxy is statistically significant (p <= 0.05).")
    else:
        print("Not enough valid folds to perform statistical test.")
    ablation_df = pd.DataFrame(fold_auprcs)
    ablation_df['metal_group'] = [groups.iloc[test_idx].iloc[0] for _, test_idx in logo.split(df[base_features], y, groups)]
    output_path = os.path.join(data_dir, "ablation_results.csv")
    ablation_df.to_csv(output_path, index=False)
    print("\nAblation results saved to " + output_path)

if __name__ == '__main__':
    main()