# filename: codebase/step_3.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
from sklearn.model_selection import LeaveOneGroupOut
import joblib
import json

def main():
    data_dir = "data"
    processed_data_path = os.path.join(data_dir, "processed_tmd_data.csv")
    targets_path = os.path.join(data_dir, "target_labels.csv")
    df_processed = pd.read_csv(processed_data_path)
    df_targets = pd.read_csv(targets_path)
    df = pd.merge(df_targets[['material_id', 'is_robust']], df_processed, on='material_id', how='inner')
    exclude_from_features = [
        'material_id', 'formula', 'metal', 'chalcogen', 'spacegroup_symbol',
        'Tc_phase', 'Tc_ref', 'Tc_exp_K', 'is_known_SC',
        'K_vrh', 'G_vrh', 'poisson_ratio', 'elastic_anisotropy', 'debye_temperature',
        'is_robust', 'is_stable', 'theoretical', 'energy_above_hull', 
        'log1p_energy_above_hull', 'formation_energy_per_atom'
    ]
    features = [c for c in df.columns if c not in exclude_from_features and not c.startswith('is_robust_')]
    X = df[features]
    y = df['is_robust']
    groups = df['metal']
    print("Number of samples for training: " + str(len(X)))
    print("Number of features: " + str(len(features)))
    print("Features used: " + ", ".join(features) + "\n")
    logo = LeaveOneGroupOut()
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=8)
    metrics = []
    for train_idx, test_idx in logo.split(X, y, groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        metal_group = groups.iloc[test_idx].iloc[0]
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        y_proba = rf.predict_proba(X_test)[:, 1]
        if len(np.unique(y_test)) > 1:
            auroc = roc_auc_score(y_test, y_proba)
            auprc = average_precision_score(y_test, y_proba)
        else:
            auroc = np.nan
            auprc = np.nan
        f1 = f1_score(y_test, y_pred, zero_division=0)
        metrics.append({
            'metal': metal_group,
            'n_samples': len(y_test),
            'auprc': auprc,
            'auroc': auroc,
            'f1': f1
        })
    print("LOCO Cross-Validation Metrics by Held-out Metal:")
    print("Metal      | Samples    | AUPRC      | AUROC      | F1 Score  ")
    print("------------------------------------------------------------")
    for row in metrics:
        auprc_str = str(round(row['auprc'], 4)) if not np.isnan(row['auprc']) else "NaN"
        auroc_str = str(round(row['auroc'], 4)) if not np.isnan(row['auroc']) else "NaN"
        f1_str = str(round(row['f1'], 4))
        metal_str = str(row['metal']).ljust(10)
        samples_str = str(row['n_samples']).ljust(10)
        auprc_str = auprc_str.ljust(10)
        auroc_str = auroc_str.ljust(10)
        f1_str = f1_str.ljust(10)
        print(metal_str + " | " + samples_str + " | " + auprc_str + " | " + auroc_str + " | " + f1_str)
    print("------------------------------------------------------------")
    mean_auprc = np.nanmean([m['auprc'] for m in metrics])
    mean_auroc = np.nanmean([m['auroc'] for m in metrics])
    mean_f1 = np.nanmean([m['f1'] for m in metrics])
    print("Macro-Avg  | -          | " + str(round(mean_auprc, 4)).ljust(10) + " | " + str(round(mean_auroc, 4)).ljust(10) + " | " + str(round(mean_f1, 4)).ljust(10) + "\n")
    metrics_df = pd.DataFrame(metrics)
    metrics_path = os.path.join(data_dir, "cv_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print("CV metrics saved to " + metrics_path)
    rf_final = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=8)
    rf_final.fit(X, y)
    model_path = os.path.join(data_dir, "rf_model.joblib")
    joblib.dump(rf_final, model_path)
    print("Trained model saved to " + model_path)
    features_path = os.path.join(data_dir, "training_features.json")
    with open(features_path, "w") as f:
        json.dump(features, f)
    print("Training features list saved to " + features_path)

if __name__ == '__main__':
    main()