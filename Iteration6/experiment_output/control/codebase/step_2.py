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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import joblib
import json

class MultiTaskTMDModel(nn.Module):
    def __init__(self, num_continuous_features, num_groups=11, group_embed_dim=4):
        super(MultiTaskTMDModel, self).__init__()
        self.group_embed = nn.Embedding(num_groups, group_embed_dim)
        self.shared = nn.Sequential(
            nn.Linear(num_continuous_features + group_embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.reg_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        self.cls_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    def forward(self, x_cont, x_group):
        group_emb = self.group_embed(x_group)
        x = torch.cat([x_cont, group_emb], dim=1)
        shared_features = self.shared(x)
        pugh_pred = self.reg_head(shared_features)
        stable_pred = self.cls_head(shared_features)
        return pugh_pred, stable_pred

def train_model(X_cont, X_group, y_reg, y_cls, epochs=1000, lr=0.005, weight_decay=1e-3):
    model = MultiTaskTMDModel(num_continuous_features=X_cont.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion_reg = nn.MSELoss()
    criterion_cls = nn.BCEWithLogitsLoss()
    X_cont_t = torch.tensor(X_cont, dtype=torch.float32)
    X_group_t = torch.tensor(X_group, dtype=torch.long)
    y_reg_t = torch.tensor(y_reg, dtype=torch.float32)
    y_cls_t = torch.tensor(y_cls, dtype=torch.float32)
    reg_mask = ~torch.isnan(y_reg_t)
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        pugh_pred, stable_pred = model(X_cont_t, X_group_t)
        if reg_mask.sum() > 0:
            pred_reg = pugh_pred[reg_mask].view(-1)
            target_reg = y_reg_t[reg_mask].view(-1)
            loss_reg = criterion_reg(pred_reg, target_reg)
        else:
            loss_reg = torch.tensor(0.0, device=X_cont_t.device)
        pred_cls = stable_pred.view(-1)
        target_cls = y_cls_t.view(-1)
        loss_cls = criterion_cls(pred_cls, target_cls)
        loss = loss_reg + 0.5 * loss_cls
        loss.backward()
        optimizer.step()
    return model

if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)
    data_path = os.path.join('data', 'processed_tmd_data.csv')
    df = pd.read_csv(data_path)
    encoded_cols = [col for col in df.columns if col.startswith('crystal_system_') or col.startswith('phase_') or col.startswith('magnetic_ordering_')]
    cont_features = ['d_band_filling', 'en_difference', 'dos_at_fermi', 'is_dos_missing', 'c_a_ratio', 'M_soc_proxy'] + encoded_cols
    if df['c_a_ratio'].isna().sum() > 0:
        df['c_a_ratio'] = df['c_a_ratio'].fillna(df['c_a_ratio'].mean())
    logo = LeaveOneGroupOut()
    groups = df['metal'].values
    all_y_true = []
    all_y_pred = []
    fold_metrics = []
    for fold, (train_idx, test_idx) in enumerate(logo.split(df, groups=groups)):
        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()
        metal_left_out = df['metal'].iloc[test_idx].iloc[0]
        dos_mean = train_df['dos_at_fermi'].mean()
        train_df['dos_at_fermi'] = train_df['dos_at_fermi'].fillna(dos_mean)
        test_df['dos_at_fermi'] = test_df['dos_at_fermi'].fillna(dos_mean)
        scaler = StandardScaler()
        X_train_cont = scaler.fit_transform(train_df[cont_features])
        X_test_cont = scaler.transform(test_df[cont_features])
        X_train_group = train_df['M_group'].values
        X_test_group = test_df['M_group'].values
        y_train_reg = train_df['pugh_ratio'].values
        y_train_cls = train_df['is_stable'].values.astype(float)
        y_test_reg = test_df['pugh_ratio'].values
        model = train_model(X_train_cont, X_train_group, y_train_reg, y_train_cls, epochs=1000, lr=0.005, weight_decay=1e-3)
        model.eval()
        with torch.no_grad():
            X_test_cont_t = torch.tensor(X_test_cont, dtype=torch.float32)
            X_test_group_t = torch.tensor(X_test_group, dtype=torch.long)
            pugh_pred, _ = model(X_test_cont_t, X_test_group_t)
            pugh_pred = pugh_pred.view(-1).numpy()
        valid_mask = ~np.isnan(y_test_reg)
        if valid_mask.sum() > 0:
            y_true_valid = y_test_reg[valid_mask]
            y_pred_valid = pugh_pred[valid_mask]
            mae = mean_absolute_error(y_true_valid, y_pred_valid)
            rmse = np.sqrt(mean_squared_error(y_true_valid, y_pred_valid))
            r2 = r2_score(y_true_valid, y_pred_valid) if len(y_true_valid) > 1 else np.nan
            fold_metrics.append({'fold': fold, 'metal_left_out': metal_left_out, 'num_samples': int(valid_mask.sum()), 'MAE': float(mae), 'RMSE': float(rmse), 'R2': float(r2)})
            all_y_true.extend(y_true_valid)
            all_y_pred.extend(y_pred_valid)
        else:
            fold_metrics.append({'fold': fold, 'metal_left_out': metal_left_out, 'num_samples': 0, 'MAE': None, 'RMSE': None, 'R2': None})
    agg_mae = mean_absolute_error(all_y_true, all_y_pred)
    agg_rmse = np.sqrt(mean_squared_error(all_y_true, all_y_pred))
    agg_r2 = r2_score(all_y_true, all_y_pred)
    results_dict = {'fold_metrics': fold_metrics, 'aggregate_metrics': {'total_samples': len(all_y_true), 'MAE': float(agg_mae), 'RMSE': float(agg_rmse), 'R2': float(agg_r2)}}
    results_path = os.path.join('data', 'logo_cv_metrics.json')
    with open(results_path, 'w') as f:
        json.dump(results_dict, f, indent=4)
    dos_mean_full = df['dos_at_fermi'].mean()
    df['dos_at_fermi'] = df['dos_at_fermi'].fillna(dos_mean_full)
    scaler_full = StandardScaler()
    X_full_cont = scaler_full.fit_transform(df[cont_features])
    X_full_group = df['M_group'].values
    y_full_reg = df['pugh_ratio'].values
    y_full_cls = df['is_stable'].values.astype(float)
    final_model = train_model(X_full_cont, X_full_group, y_full_reg, y_full_cls, epochs=1000, lr=0.005, weight_decay=1e-3)
    if agg_r2 < 0.1:
        rf_all_y_true = []
        rf_all_y_pred = []
        rf_fold_metrics = []
        for fold, (train_idx, test_idx) in enumerate(logo.split(df, groups=groups)):
            train_df = df.iloc[train_idx].copy()
            test_df = df.iloc[test_idx].copy()
            metal_left_out = df['metal'].iloc[test_idx].iloc[0]
            dos_mean = train_df['dos_at_fermi'].mean()
            train_df['dos_at_fermi'] = train_df['dos_at_fermi'].fillna(dos_mean)
            test_df['dos_at_fermi'] = test_df['dos_at_fermi'].fillna(dos_mean)
            train_df_reg = train_df.dropna(subset=['pugh_ratio'])
            X_train = train_df_reg[cont_features + ['M_group']]
            y_train = train_df_reg['pugh_ratio']
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X_train, y_train)
            test_df_reg = test_df.dropna(subset=['pugh_ratio'])
            if len(test_df_reg) > 0:
                X_test = test_df_reg[cont_features + ['M_group']]
                y_test = test_df_reg['pugh_ratio']
                y_pred = rf.predict(X_test)
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred) if len(y_test) > 1 else np.nan
                rf_fold_metrics.append({'fold': fold, 'metal_left_out': metal_left_out, 'num_samples': len(y_test), 'MAE': float(mae), 'RMSE': float(rmse), 'R2': float(r2)})
                rf_all_y_true.extend(y_test)
                rf_all_y_pred.extend(y_pred)
            else:
                rf_fold_metrics.append({'fold': fold, 'metal_left_out': metal_left_out, 'num_samples': 0, 'MAE': None, 'RMSE': None, 'R2': None})
        rf_agg_mae = mean_absolute_error(rf_all_y_true, rf_all_y_pred)
        rf_agg_rmse = np.sqrt(mean_squared_error(rf_all_y_true, rf_all_y_pred))
        rf_agg_r2 = r2_score(rf_all_y_true, rf_all_y_pred)
        results_dict = {'fold_metrics': rf_fold_metrics, 'aggregate_metrics': {'total_samples': len(rf_all_y_true), 'MAE': float(rf_agg_mae), 'RMSE': float(rf_agg_rmse), 'R2': float(rf_agg_r2)}, 'model_type': 'RandomForest'}
        with open(results_path, 'w') as f:
            json.dump(results_dict, f, indent=4)
        df_reg = df.dropna(subset=['pugh_ratio'])
        X_full_reg = df_reg[cont_features + ['M_group']]
        y_full_reg = df_reg['pugh_ratio']
        final_rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
        final_rf_reg.fit(X_full_reg, y_full_reg)
        X_full_cls = df[cont_features + ['M_group']]
        y_full_cls = df['is_stable'].astype(int)
        final_rf_cls = RandomForestClassifier(n_estimators=100, random_state=42)
        final_rf_cls.fit(X_full_cls, y_full_cls)
        model_path = os.path.join('data', 'tmd_model.joblib')
        joblib.dump({'model_reg': final_rf_reg, 'model_cls': final_rf_cls, 'features': cont_features + ['M_group'], 'dos_mean_full': dos_mean_full, 'model_type': 'RandomForest'}, model_path)
    else:
        results_dict['model_type'] = 'NeuralNetwork'
        with open(results_path, 'w') as f:
            json.dump(results_dict, f, indent=4)
        model_path = os.path.join('data', 'tmd_model.pth')
        torch.save({'model_state_dict': final_model.state_dict(), 'scaler_mean': scaler_full.mean_, 'scaler_scale': scaler_full.scale_, 'cont_features': cont_features, 'dos_mean_full': dos_mean_full, 'model_type': 'NeuralNetwork'}, model_path)