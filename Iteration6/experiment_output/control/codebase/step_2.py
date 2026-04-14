# filename: codebase/step_2.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score, accuracy_score
import joblib
import json

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)

class MultiTaskNN(nn.Module):
    def __init__(self, num_continuous, num_binary, embedding_dim=2):
        super(MultiTaskNN, self).__init__()
        self.group_emb = nn.Embedding(2, embedding_dim)
        input_dim = num_continuous + num_binary + embedding_dim
        self.shared = nn.Sequential(nn.Linear(input_dim, 32), nn.ReLU(), nn.Dropout(0.1), nn.Linear(32, 16), nn.ReLU(), nn.Dropout(0.1))
        self.reg_head = nn.Linear(16, 1)
        self.clf_head = nn.Linear(16, 1)
    def forward(self, x_cont, x_bin, x_group):
        emb = self.group_emb(x_group)
        x = torch.cat([x_cont, x_bin, emb], dim=1)
        shared_rep = self.shared(x)
        reg_out = self.reg_head(shared_rep)
        clf_out = self.clf_head(shared_rep)
        return reg_out, clf_out

def train_nn(X_cont, X_bin, X_group, y_reg, y_clf, mask_reg, epochs=300, lr=0.005, weight_decay=1e-3):
    set_seed(42)
    model = MultiTaskNN(X_cont.shape[1], X_bin.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion_reg = nn.MSELoss()
    criterion_clf = nn.BCEWithLogitsLoss()
    X_cont_t = torch.tensor(X_cont, dtype=torch.float32)
    X_bin_t = torch.tensor(X_bin, dtype=torch.float32)
    X_group_t = torch.tensor(X_group, dtype=torch.long)
    y_reg_t = torch.tensor(y_reg, dtype=torch.float32).unsqueeze(1)
    y_clf_t = torch.tensor(y_clf, dtype=torch.float32).unsqueeze(1)
    mask_reg_t = torch.tensor(mask_reg, dtype=torch.bool)
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

def main():
    data_path = os.path.join("data", "processed_tmd_data.csv")
    df = pd.read_csv(data_path)
    continuous_features = ['d_band_filling', 'en_difference', 'dos_at_fermi', 'c_a_ratio', 'M_soc_proxy']
    binary_features = ['is_dos_missing'] + [c for c in df.columns if c.startswith('crystal_system_') or c.startswith('phase_') or c.startswith('magnetic_ordering_')]
    df['M_group_bin'] = (df['M_group'] >= 8).astype(int)
    target_reg = 'pugh_ratio'
    target_clf = 'is_stable'
    groups = df['metal'].values
    logo = LeaveOneGroupOut()
    y_reg_pred_nn = np.full(len(df), np.nan)
    y_clf_pred_nn = np.full(len(df), np.nan)
    for train_idx, test_idx in logo.split(df, groups=groups):
        df_train = df.iloc[train_idx].copy()
        df_test = df.iloc[test_idx].copy()
        dos_mean = df_train['dos_at_fermi'].mean()
        df_train['dos_at_fermi'] = df_train['dos_at_fermi'].fillna(dos_mean)
        df_test['dos_at_fermi'] = df_test['dos_at_fermi'].fillna(dos_mean)
        scaler = StandardScaler()
        X_cont_train = scaler.fit_transform(df_train[continuous_features])
        X_cont_test = scaler.transform(df_test[continuous_features])
        X_bin_train = df_train[binary_features].values
        X_bin_test = df_test[binary_features].values
        X_group_train = df_train['M_group_bin'].values
        X_group_test = df_test['M_group_bin'].values
        y_reg_train = df_train[target_reg].fillna(0).values
        mask_reg_train = df_train[target_reg].notna().values
        y_clf_train = df_train[target_clf].astype(float).values
        model = train_nn(X_cont_train, X_bin_train, X_group_train, y_reg_train, y_clf_train, mask_reg_train, epochs=300, lr=0.005)
        model.eval()
        with torch.no_grad():
            reg_out, clf_out = model(torch.tensor(X_cont_test, dtype=torch.float32), torch.tensor(X_bin_test, dtype=torch.float32), torch.tensor(X_group_test, dtype=torch.long))
        y_reg_pred_nn[test_idx] = reg_out.numpy().flatten()
        y_clf_pred_nn[test_idx] = torch.sigmoid(clf_out).numpy().flatten()
    mask_reg_all = df[target_reg].notna()
    y_reg_true = df.loc[mask_reg_all, target_reg].values
    y_reg_pred = y_reg_pred_nn[mask_reg_all]
    nn_mae = mean_absolute_error(y_reg_true, y_reg_pred)
    nn_rmse = np.sqrt(mean_squared_error(y_reg_true, y_reg_pred))
    nn_r2 = r2_score(y_reg_true, y_reg_pred)
    y_clf_true = df[target_clf].astype(float).values
    y_clf_pred = y_clf_pred_nn
    nn_auc = roc_auc_score(y_clf_true, y_clf_pred)
    nn_acc = accuracy_score(y_clf_true, (y_clf_pred > 0.5).astype(int))
    use_rf = False
    if nn_r2 < 0.1:
        use_rf = True
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
        y_reg_pred_rf = np.full(len(df), np.nan)
        y_clf_pred_rf = np.full(len(df), np.nan)
        for train_idx, test_idx in logo.split(df, groups=groups):
            df_train = df.iloc[train_idx].copy()
            df_test = df.iloc[test_idx].copy()
            dos_mean = df_train['dos_at_fermi'].mean()
            df_train['dos_at_fermi'] = df_train['dos_at_fermi'].fillna(dos_mean)
            df_test['dos_at_fermi'] = df_test['dos_at_fermi'].fillna(dos_mean)
            X_train = pd.concat([df_train[continuous_features], df_train[binary_features], df_train[['M_group_bin']]], axis=1)
            X_test = pd.concat([df_test[continuous_features], df_test[binary_features], df_test[['M_group_bin']]], axis=1)
            mask_reg_train = df_train[target_reg].notna()
            rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_reg.fit(X_train[mask_reg_train], df_train.loc[mask_reg_train, target_reg])
            y_reg_pred_rf[test_idx] = rf_reg.predict(X_test)
            rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_clf.fit(X_train, df_train[target_clf])
            y_clf_pred_rf[test_idx] = rf_clf.predict_proba(X_test)[:, 1]
        y_reg_pred = y_reg_pred_rf[mask_reg_all]
        rf_mae = mean_absolute_error(y_reg_true, y_reg_pred)
        rf_rmse = np.sqrt(mean_squared_error(y_reg_true, y_reg_pred))
        rf_r2 = r2_score(y_reg_true, y_reg_pred)
        y_clf_pred = y_clf_pred_rf
        rf_auc = roc_auc_score(y_clf_true, y_clf_pred)
        rf_acc = accuracy_score(y_clf_true, (y_clf_pred > 0.5).astype(int))
        final_mae, final_rmse, final_r2 = rf_mae, rf_rmse, rf_r2
        final_auc, final_acc = rf_auc, rf_acc
        final_y_reg_pred = y_reg_pred_rf
        final_y_clf_pred = y_clf_pred_rf
    else:
        final_mae, final_rmse, final_r2 = nn_mae, nn_rmse, nn_r2
        final_auc, final_acc = nn_auc, nn_acc
        final_y_reg_pred = y_reg_pred_nn
        final_y_clf_pred = y_clf_pred_nn
    df['cv_pred_pugh_ratio'] = final_y_reg_pred
    df['cv_pred_is_stable'] = final_y_clf_pred
    df.to_csv(data_path, index=False)
    metrics = {'regression': {'MAE': final_mae, 'RMSE': final_rmse, 'R2': final_r2}, 'classification': {'AUC': final_auc, 'Accuracy': final_acc}, 'model_type': 'RandomForest' if use_rf else 'NeuralNetwork'}
    metrics_path = os.path.join("data", "logo_cv_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    df_full = df.copy()
    dos_mean_full = df_full['dos_at_fermi'].mean()
    df_full['dos_at_fermi'] = df_full['dos_at_fermi'].fillna(dos_mean_full)
    model_save_path = os.path.join("data", "tmd_model.joblib")
    if use_rf:
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
        X_full = pd.concat([df_full[continuous_features], df_full[binary_features], df_full[['M_group_bin']]], axis=1)
        mask_reg_full = df_full[target_reg].notna()
        rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_reg.fit(X_full[mask_reg_full], df_full.loc[mask_reg_full, target_reg])
        rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_clf.fit(X_full, df_full[target_clf])
        final_model = {'reg': rf_reg, 'clf': rf_clf, 'dos_mean': dos_mean_full, 'features': list(X_full.columns), 'type': 'rf'}
        joblib.dump(final_model, model_save_path)
    else:
        scaler_full = StandardScaler()
        X_cont_full = scaler_full.fit_transform(df_full[continuous_features])
        X_bin_full = df_full[binary_features].values
        X_group_full = df_full['M_group_bin'].values
        y_reg_full = df_full[target_reg].fillna(0).values
        mask_reg_full = df_full[target_reg].notna().values
        y_clf_full = df_full[target_clf].astype(float).values
        final_model_nn = train_nn(X_cont_full, X_bin_full, X_group_full, y_reg_full, y_clf_full, mask_reg_full, epochs=300, lr=0.005)
        final_model = {'model_state': final_model_nn.state_dict(), 'scaler': scaler_full, 'dos_mean': dos_mean_full, 'continuous_features': continuous_features, 'binary_features': binary_features, 'type': 'nn'}
        joblib.dump(final_model, model_save_path)

if __name__ == '__main__':
    main()