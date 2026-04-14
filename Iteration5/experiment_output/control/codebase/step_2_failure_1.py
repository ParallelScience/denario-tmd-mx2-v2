# filename: codebase/step_2.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, roc_auc_score

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class MultiTaskNet(nn.Module):
    def __init__(self, input_dim):
        super(MultiTaskNet, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.reg_head = nn.Linear(32, 1)
        self.clf_head = nn.Linear(32, 1)
        self.log_var_reg = nn.Parameter(torch.zeros(1))
        self.log_var_clf = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        shared_features = self.shared(x)
        reg_out = self.reg_head(shared_features)
        clf_out = torch.sigmoid(self.clf_head(shared_features))
        return reg_out, clf_out

def multi_task_loss(reg_pred, reg_target, clf_pred, clf_target, log_var_reg, log_var_clf, reg_mask):
    mse_loss_fn = nn.MSELoss(reduction='none')
    mse = mse_loss_fn(reg_pred.view(-1), reg_target.view(-1))
    mse = (mse * reg_mask.view(-1)).sum() / (reg_mask.sum() + 1e-8)
    bce_loss_fn = nn.BCELoss()
    bce = bce_loss_fn(clf_pred.view(-1), clf_target.view(-1))
    loss = 0.5 * torch.exp(-log_var_reg) * mse + 0.5 * torch.exp(-log_var_clf) * bce + 0.5 * (log_var_reg + log_var_clf)
    return loss, mse, bce

def elastic_net_loss(model, l1_ratio=0.5, alpha=1e-4, protected_indices=[]):
    l1_loss = 0.0
    l2_loss = 0.0
    for name, param in model.named_parameters():
        if 'shared.0.weight' in name:
            mask = torch.ones_like(param)
            for idx in protected_indices:
                mask[:, idx] = 0.0
            l1_loss += torch.sum(torch.abs(param * mask))
            l2_loss += torch.sum((param * mask) ** 2)
        elif 'weight' in name and 'shared' in name:
            l1_loss += torch.sum(torch.abs(param))
            l2_loss += torch.sum(param ** 2)
    return alpha * (l1_ratio * l1_loss + (1 - l1_ratio) * l2_loss)

def train_model(X_train, y_reg_train, y_clf_train, reg_mask_train, protected_indices, epochs=300, lr=0.005, batch_size=32, device='cpu'):
    model = MultiTaskNet(input_dim=X_train.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    dataset = TensorDataset(X_train, y_reg_train, y_clf_train, reg_mask_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y_reg, batch_y_clf, batch_reg_mask in dataloader:
            batch_X = batch_X.to(device)
            batch_y_reg = batch_y_reg.to(device)
            batch_y_clf = batch_y_clf.to(device)
            batch_reg_mask = batch_reg_mask.to(device)
            optimizer.zero_grad()
            reg_pred, clf_pred = model(batch_X)
            loss, mse, bce = multi_task_loss(reg_pred, batch_y_reg, clf_pred, batch_y_clf, model.log_var_reg, model.log_var_clf, batch_reg_mask)
            reg_loss = elastic_net_loss(model, l1_ratio=0.5, alpha=1e-4, protected_indices=protected_indices)
            total_loss = loss + reg_loss
            total_loss.backward()
            optimizer.step()
    return model

def main():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device: " + str(device))
    known_df = pd.read_csv('data/known_df_processed.csv')
    unknown_df = pd.read_csv('data/unknown_df_processed.csv')
    with open('data/feature_list.json', 'r') as f:
        feature_list = json.load(f)
    full_df = pd.concat([known_df, unknown_df], ignore_index=True)
    X = full_df[feature_list].values
    y_reg = full_df['G_vrh_norm'].values
    y_clf = full_df['is_stable'].values
    groups = full_df['M_group'].values
    energy_above_hull = full_df['energy_above_hull'].values
    reg_mask = ~np.isnan(y_reg)
    y_reg_safe = np.nan_to_num(y_reg, nan=0.0)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_reg_tensor = torch.tensor(y_reg_safe, dtype=torch.float32)
    y_clf_tensor = torch.tensor(y_clf, dtype=torch.float32)
    reg_mask_tensor = torch.tensor(reg_mask, dtype=torch.float32)
    protected_features = ['d_band_filling', 'en_difference']
    protected_indices = [feature_list.index(f) for f in protected_features if f in feature_list]
    print("\n=== Starting LOGO Cross-Validation ===")
    logo = LeaveOneGroupOut()
    fold = 1
    all_metrics = []
    for train_idx, test_idx in logo.split(X_tensor, groups=groups):
        X_train, X_test = X_tensor[train_idx], X_tensor[test_idx]
        y_reg_train, y_reg_test = y_reg_tensor[train_idx], y_reg_tensor[test_idx]
        y_clf_train, y_clf_test = y_clf_tensor[train_idx], y_clf_tensor[test_idx]
        reg_mask_train, reg_mask_test = reg_mask_tensor[train_idx], reg_mask_tensor[test_idx]
        model = train_model(X_train, y_reg_train, y_clf_train, reg_mask_train, protected_indices, epochs=200, lr=0.005, device=device)
        model.eval()
        with torch.no_grad():
            reg_pred, clf_pred = model(X_test.to(device))
            reg_pred = reg_pred.cpu().view(-1).numpy()
            clf_pred = clf_pred.cpu().view(-1).numpy()
        test_reg_mask = reg_mask_test.numpy().astype(bool)
        if test_reg_mask.sum() > 0:
            y_reg_true = y_reg_test.numpy()[test_reg_mask]
            y_reg_p = reg_pred[test_reg_mask]
            mae = mean_absolute_error(y_reg_true, y_reg_p)
            rmse = np.sqrt(mean_squared_error(y_reg_true, y_reg_p))
            r2 = r2_score(y_reg_true, y_reg_p) if len(y_reg_true) > 1 else np.nan
        else:
            mae, rmse, r2 = np.nan, np.nan, np.nan
        y_clf_true = y_clf_test.numpy()
        acc = accuracy_score(y_clf_true, (clf_pred > 0.5).astype(int))
        roc_auc = roc_auc_score(y_clf_true, clf_pred) if len(np.unique(y_clf_true)) > 1 else np.nan
        eah_test = energy_above_hull[test_idx]
        corr = np.corrcoef(reg_pred, eah_test)[0, 1] if len(eah_test) > 1 and np.std(reg_pred) > 1e-6 and np.std(eah_test) > 1e-6 else np.nan
        group_val = groups[test_idx][0]
        print("Fold " + str(fold) + " (Group " + str(group_val) + "):")
        print("  Reg: MAE=" + str(round(mae, 4)) + ", RMSE=" + str(round(rmse, 4)) + ", R2=" + str(round(r2, 4)))
        print("  Clf: Acc=" + str(round(acc, 4)) + ", ROC-AUC=" + str(round(roc_auc, 4)))
        print("  Corr(G_vrh_pred, E_hull)=" + str(round(corr, 4)))
        all_metrics.append({'fold': fold, 'group': group_val, 'mae': mae, 'rmse': rmse, 'r2': r2, 'acc': acc, 'roc_auc': roc_auc, 'corr': corr})
        fold += 1
    print("\n=== Average CV Metrics ===")
    avg_mae = np.nanmean([m['mae'] for m in all_metrics])
    avg_rmse = np.nanmean([m['rmse'] for m in all_metrics])
    avg_r2 = np.nanmean([m['r2'] for m in all_metrics])
    avg_acc = np.nanmean([m['acc'] for m in all_metrics])
    avg_roc_auc = np.nanmean([m['roc_auc'] for m in all_metrics])
    avg_corr = np.nanmean([m['corr'] for m in all_metrics])
    print("  Reg: MAE=" + str(round(avg_mae, 4)) + ", RMSE=" + str(round(avg_rmse, 4)) + ", R2=" + str(round(avg_r2, 4)))
    print("  Clf: Acc=" + str(round(avg_acc, 4)) + ", ROC-AUC=" + str(round(avg_roc_auc, 4)))
    print("  Corr(G_vrh_pred, E_hull)=" + str(round(avg_corr, 4)))
    print("\n=== Retraining Final Model on Full Dataset ===")
    final_model = train_model(X_tensor, y_reg_tensor, y_clf_tensor, reg_mask_tensor, protected_indices, epochs=300, lr=0.005, device=device)
    sigma_1 = torch.exp(0.5 * final_model.log_var_reg).item()
    sigma_2 = torch.exp(0.5 * final_model.log_var_clf).item()
    print("\nLearned Uncertainty Weights:")
    print("  sigma_1 (Regression): " + str(round(sigma_1, 4)))
    print("  sigma_2 (Classification): " + str(round(sigma_2, 4)))
    torch.save(final_model.state_dict(), 'data/final_multitask_model.pth')
    weights = {'sigma_1': sigma_1, 'sigma_2': sigma_2, 'log_var_reg': final_model.log_var_reg.item(), 'log_var_clf': final_model.log_var_clf.item()}
    with open('data/uncertainty_weights.json', 'w') as f:
        json.dump(weights, f)
    print("\nModel saved to data/final_multitask_model.pth")
    print("Uncertainty weights saved to data/uncertainty_weights.json")

if __name__ == '__main__':
    main()