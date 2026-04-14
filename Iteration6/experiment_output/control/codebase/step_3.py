# filename: codebase/step_3.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np
import json
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from step_2 import MultiTaskNN
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
def train_nn_with_seed(X_cont, X_bin, X_group, y_reg, y_clf, mask_reg, seed=42, epochs=300, lr=0.005, weight_decay=1e-3):
    set_seed(seed)
    model = MultiTaskNN(X_cont.shape[1], X_bin.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion_reg = nn.MSELoss()
    criterion_clf = nn.BCEWithLogitsLoss()
    X_cont_t = torch.tensor(X_cont, dtype=torch.float32).to(device)
    X_bin_t = torch.tensor(X_bin, dtype=torch.float32).to(device)
    X_group_t = torch.tensor(X_group, dtype=torch.long).to(device)
    y_reg_t = torch.tensor(y_reg, dtype=torch.float32).unsqueeze(1).to(device)
    y_clf_t = torch.tensor(y_clf, dtype=torch.float32).unsqueeze(1).to(device)
    mask_reg_t = torch.tensor(mask_reg, dtype=torch.bool).to(device)
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        reg_out, clf_out = model(X_cont_t, X_bin_t, X_group_t)
        loss_clf = criterion_clf(clf_out, y_clf_t)
        if mask_reg_t.sum() > 0:
            loss_reg = criterion_reg(reg_out[mask_reg_t], y_reg_t[mask_reg_t])
            loss = loss_reg + loss_clf
        else:
            loss = loss_clf
        loss.backward()
        optimizer.step()
    return model
def evaluate_cv(df, continuous_features, binary_features, model_type):
    groups = df['metal'].values
    logo = LeaveOneGroupOut()
    y_true = []
    y_pred = []
    for train_idx, test_idx in logo.split(df, groups=groups):
        df_train = df.iloc[train_idx].copy()
        df_test = df.iloc[test_idx].copy()
        dos_mean = df_train['dos_at_fermi'].mean()
        df_train['dos_at_fermi'] = df_train['dos_at_fermi'].fillna(dos_mean)
        df_test['dos_at_fermi'] = df_test['dos_at_fermi'].fillna(dos_mean)
        mask_reg_train = df_train['pugh_ratio'].notna()
        mask_reg_test = df_test['pugh_ratio'].notna()
        if mask_reg_test.sum() == 0:
            continue
        if model_type == 'RandomForest':
            X_train = pd.concat([df_train[continuous_features], df_train[binary_features], df_train[['M_group_bin']]], axis=1)
            X_test = pd.concat([df_test[continuous_features], df_test[binary_features], df_test[['M_group_bin']]], axis=1)
            rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_reg.fit(X_train[mask_reg_train], df_train.loc[mask_reg_train, 'pugh_ratio'])
            preds = rf_reg.predict(X_test[mask_reg_test])
        else:
            scaler = StandardScaler()
            X_cont_train = scaler.fit_transform(df_train[continuous_features])
            X_cont_test = scaler.transform(df_test[continuous_features])
            X_bin_train = df_train[binary_features].values
            X_bin_test = df_test[binary_features].values
            X_group_train = df_train['M_group_bin'].values
            X_group_test = df_test['M_group_bin'].values
            y_reg_train = df_train['pugh_ratio'].fillna(0).values
            mask_reg_train_vals = mask_reg_train.values
            y_clf_train = df_train['is_stable'].astype(float).values
            model = train_nn_with_seed(X_cont_train, X_bin_train, X_group_train, y_reg_train, y_clf_train, mask_reg_train_vals, seed=42, epochs=300, lr=0.005)
            model.eval()
            with torch.no_grad():
                reg_out, _ = model(torch.tensor(X_cont_test, dtype=torch.float32).to(device), torch.tensor(X_bin_test, dtype=torch.float32).to(device), torch.tensor(X_group_test, dtype=torch.long).to(device))
            preds = reg_out.cpu().numpy().flatten()[mask_reg_test.values]
        y_true.extend(df_test.loc[mask_reg_test, 'pugh_ratio'].values)
        y_pred.extend(preds)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mae, r2
def main():
    data_path = os.path.join("data", "processed_tmd_data.csv")
    df = pd.read_csv(data_path)
    metrics_path = os.path.join("data", "logo_cv_metrics.json")
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    model_type = metrics.get("model_type", "RandomForest")
    continuous_features = ['d_band_filling', 'en_difference', 'dos_at_fermi', 'c_a_ratio', 'M_soc_proxy']
    binary_features_with = ['is_dos_missing'] + [c for c in df.columns if c.startswith('crystal_system_') or c.startswith('phase_') or c.startswith('magnetic_ordering_')]
    binary_features_without = [c for c in df.columns if c.startswith('crystal_system_') or c.startswith('phase_') or c.startswith('magnetic_ordering_')]
    if 'M_group_bin' not in df.columns:
        df['M_group_bin'] = (df['M_group'] >= 8).astype(int)
    print("Evaluating sensitivity to missing DOS indicator using " + model_type + "...")
    mae_with, r2_with = evaluate_cv(df, continuous_features, binary_features_with, model_type)
    mae_without, r2_without = evaluate_cv(df, continuous_features, binary_features_without, model_type)
    print("\n--- Sensitivity Analysis on Missing Data ---")
    print(f"{'Model Configuration':<30} | {'MAE':<10} | {'R2':<10}")
    print("-" * 56)
    print(f"{'With is_dos_missing':<30} | {mae_with:<10.4f} | {r2_with:<10.4f}")
    print(f"{'Without is_dos_missing':<30} | {mae_without:<10.4f} | {r2_without:<10.4f}")
    print("-" * 56)
    print("\nTraining ensemble of 10 " + model_type + " models...")
    ensemble_models = []
    predictions = np.zeros((len(df), 10))
    df_full = df.copy()
    dos_mean_full = df_full['dos_at_fermi'].mean()
    df_full['dos_at_fermi'] = df_full['dos_at_fermi'].fillna(dos_mean_full)
    mask_reg_full = df_full['pugh_ratio'].notna()
    if model_type == 'RandomForest':
        X_full = pd.concat([df_full[continuous_features], df_full[binary_features_with], df_full[['M_group_bin']]], axis=1)
        y_reg_full = df_full.loc[mask_reg_full, 'pugh_ratio']
        for i in range(10):
            rf_reg = RandomForestRegressor(n_estimators=100, random_state=i)
            rf_reg.fit(X_full[mask_reg_full], y_reg_full)
            preds = rf_reg.predict(X_full)
            predictions[:, i] = preds
            ensemble_models.append(rf_reg)
        scaler_full = None
    else:
        scaler_full = StandardScaler()
        X_cont_full = scaler_full.fit_transform(df_full[continuous_features])
        X_bin_full = df_full[binary_features_with].values
        X_group_full = df_full['M_group_bin'].values
        y_reg_full = df_full['pugh_ratio'].fillna(0).values
        mask_reg_full_vals = mask_reg_full.values
        y_clf_full = df_full['is_stable'].astype(float).values
        for i in range(10):
            model = train_nn_with_seed(X_cont_full, X_bin_full, X_group_full, y_reg_full, y_clf_full, mask_reg_full_vals, seed=i, epochs=300, lr=0.005)
            model.eval()
            with torch.no_grad():
                reg_out, _ = model(torch.tensor(X_cont_full, dtype=torch.float32).to(device), torch.tensor(X_bin_full, dtype=torch.float32).to(device), torch.tensor(X_group_full, dtype=torch.long).to(device))
            preds = reg_out.cpu().numpy().flatten()
            predictions[:, i] = preds
            ensemble_models.append(model.cpu().state_dict())
    mean_preds = predictions.mean(axis=1)
    std_preds = predictions.std(axis=1)
    df_uncertainty = pd.DataFrame({'material_id': df['material_id'], 'mean_pred_pugh_ratio': mean_preds, 'std_pred_pugh_ratio': std_preds})
    uncertainty_path = os.path.join("data", "uncertainty_metrics.csv")
    df_uncertainty.to_csv(uncertainty_path, index=False)
    print("\nEnsemble predictions saved to: " + uncertainty_path)
    ensemble_save_path = os.path.join("data", "ensemble_models.joblib")
    joblib.dump({'models': ensemble_models, 'model_type': model_type, 'dos_mean': dos_mean_full, 'features': continuous_features + binary_features_with + ['M_group_bin'] if model_type == 'RandomForest' else None, 'continuous_features': continuous_features if model_type != 'RandomForest' else None, 'binary_features': binary_features_with if model_type != 'RandomForest' else None, 'scaler': scaler_full}, ensemble_save_path)
    print("Ensemble models saved to: " + ensemble_save_path)
if __name__ == '__main__':
    main()