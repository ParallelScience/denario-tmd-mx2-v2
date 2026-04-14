# filename: codebase/step_3.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE, RandomOverSampler
import joblib

mpl.rcParams['text.usetex'] = False

if __name__ == '__main__':
    data_dir = 'data/'
    df = pd.read_csv(os.path.join(data_dir, 'training_dataset_step2.csv'))
    with open(os.path.join(data_dir, 'selected_features.txt'), 'r') as f:
        selected_features = [line.strip() for line in f.readlines() if line.strip()]
    X = df[selected_features]
    y = df['is_viable'].astype(int)
    groups = df['M_group']
    print('Sample counts per M_group:')
    group_counts = groups.value_counts().sort_index()
    for g, count in group_counts.items():
        print('  Group ' + str(g) + ': ' + str(count) + ' samples')
    logo = LeaveOneGroupOut()
    roc_aucs = []
    bal_accs = []
    agg_cm = np.zeros((2, 2), dtype=int)
    print('\nStarting LOGO Cross-Validation...')
    fold = 1
    for train_idx, test_idx in logo.split(X, y, groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        test_group = groups.iloc[test_idx].iloc[0]
        if len(np.unique(y_train)) > 1:
            min_class_count = y_train.value_counts().min()
            k_neighbors = min(5, min_class_count - 1)
            if k_neighbors > 0:
                smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
                X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
            else:
                ros = RandomOverSampler(random_state=42)
                X_train_res, y_train_res = ros.fit_resample(X_train, y_train)
        else:
            X_train_res, y_train_res = X_train, y_train
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train_res, y_train_res)
        y_pred = rf.predict(X_test)
        if len(np.unique(y_test)) > 1:
            y_prob = rf.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_prob)
        else:
            auc = np.nan
        b_acc = balanced_accuracy_score(y_test, y_pred)
        roc_aucs.append(auc)
        bal_accs.append(b_acc)
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        agg_cm += cm
        auc_str = str(round(auc, 4)) if not np.isnan(auc) else 'NaN'
        print('  Fold ' + str(fold) + ' (Left out Group ' + str(test_group) + '): ROC-AUC = ' + auc_str + ', Balanced Acc = ' + str(round(b_acc, 4)))
        fold += 1
    valid_aucs = [a for a in roc_aucs if not np.isnan(a)]
    avg_auc = np.mean(valid_aucs) if valid_aucs else np.nan
    avg_b_acc = np.mean(bal_accs)
    print('\nCross-Validation Results:')
    print('  Average ROC-AUC: ' + str(round(avg_auc, 4)))
    print('  Average Balanced Accuracy: ' + str(round(avg_b_acc, 4)))
    min_class_count_all = y.value_counts().min()
    k_neighbors_all = min(5, min_class_count_all - 1)
    if k_neighbors_all > 0:
        smote_all = SMOTE(random_state=42, k_neighbors=k_neighbors_all)
        X_res, y_res = smote_all.fit_resample(X, y)
    else:
        ros_all = RandomOverSampler(random_state=42)
        X_res, y_res = ros_all.fit_resample(X, y)
    final_rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    final_rf.fit(X_res, y_res)
    model_path = os.path.join(data_dir, 'rf_model_step3.joblib')
    joblib.dump(final_rf, model_path)
    print('\nFinal model trained and saved to ' + model_path)
    plt.figure(figsize=(6, 5))
    sns.heatmap(agg_cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=['Not Viable (0)', 'Viable (1)'], yticklabels=['Not Viable (0)', 'Viable (1)'])
    plt.title('Aggregated Confusion Matrix (LOGO CV)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    timestamp = int(time.time())
    plot_filename = os.path.join(data_dir, 'aggregated_confusion_matrix_3_' + str(timestamp) + '.png')
    plt.savefig(plot_filename, dpi=300)
    print('Aggregated confusion matrix plot saved to ' + plot_filename)