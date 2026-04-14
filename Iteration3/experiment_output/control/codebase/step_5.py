# filename: codebase/step_5.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
import joblib
import json
import time

def main():
    plt.rcParams['text.usetex'] = False
    data_dir = 'data'
    processed_data_path = os.path.join(data_dir, 'processed_tmd_data.csv')
    targets_path = os.path.join(data_dir, 'target_labels.csv')
    features_path = os.path.join(data_dir, 'training_features.json')
    model_path = os.path.join(data_dir, 'rf_model.joblib')
    df_processed = pd.read_csv(processed_data_path)
    df_targets = pd.read_csv(targets_path)
    with open(features_path, 'r') as f:
        training_features = json.load(f)
    rf_model = joblib.load(model_path)
    train_mids = set(df_targets['material_id'])
    train_df = df_processed[df_processed['material_id'].isin(train_mids)].copy()
    meta_df = df_processed[~df_processed['material_id'].isin(train_mids)].copy()
    one_hot_prefixes = ['phase_', 'crystal_system_', 'magnetic_ordering_']
    cont_features = [f for f in training_features if not any(f.startswith(p) for p in one_hot_prefixes)]
    X_train_cont = train_df[cont_features].values
    X_meta_cont = meta_df[cont_features].values
    knn_train = NearestNeighbors(n_neighbors=6, metric='euclidean')
    knn_train.fit(X_train_cont)
    distances_train, _ = knn_train.kneighbors(X_train_cont)
    mean_dist_train = distances_train[:, 1:].mean(axis=1)
    threshold_95 = np.percentile(mean_dist_train, 95)
    knn_meta = NearestNeighbors(n_neighbors=5, metric='euclidean')
    knn_meta.fit(X_train_cont)
    distances_meta, _ = knn_meta.kneighbors(X_meta_cont)
    mean_dist_meta = distances_meta.mean(axis=1)
    meta_df['knn_distance'] = mean_dist_meta
    meta_df['is_ood'] = mean_dist_meta > threshold_95
    X_meta_full = meta_df[training_features]
    meta_df['pred_robust'] = rf_model.predict(X_meta_full)
    meta_df['pred_proba'] = rf_model.predict_proba(X_meta_full)[:, 1]
    ood_df = meta_df[meta_df['is_ood']]
    print('95th percentile threshold for internal distance: ' + str(round(threshold_95, 4)))
    print('Number of OOD candidates: ' + str(len(ood_df)))
    print('\nOOD Candidates (ID | Formula | Distance):')
    for _, row in ood_df.iterrows():
        print(str(row['material_id']) + ' | ' + str(row['formula']) + ' | ' + str(round(row['knn_distance'], 4)))
    print('\n')
    train_df['plot_category'] = 'Stable'
    def get_meta_category(row):
        if row['is_ood']:
            return 'Metastable-OOD'
        elif row['pred_robust'] == 1:
            return 'Metastable-Robust'
        else:
            return 'Metastable-NonRobust'
    meta_df['plot_category'] = meta_df.apply(get_meta_category, axis=1)
    all_df = pd.concat([train_df, meta_df], ignore_index=True)
    X_all_cont = all_df[cont_features].values
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_results = tsne.fit_transform(X_all_cont)
    all_df['tsne_1'] = tsne_results[:, 0]
    all_df['tsne_2'] = tsne_results[:, 1]
    plt.figure(figsize=(10, 8))
    palette = {'Stable': '#1f77b4', 'Metastable-Robust': '#2ca02c', 'Metastable-OOD': '#d62728', 'Metastable-NonRobust': '#ff7f0e'}
    sns.scatterplot(x='tsne_1', y='tsne_2', hue='plot_category', palette=palette, data=all_df, alpha=0.8, s=60, edgecolor='k')
    plt.title('t-SNE Projection of TMD Feature Space')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend(title='Category')
    plt.tight_layout()
    timestamp = int(time.time())
    plot_filename = 'tsne_projection_5_' + str(timestamp) + '.png'
    plot_path = os.path.join(data_dir, plot_filename)
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print('t-SNE plot saved to ' + plot_path)
    ood_flags_df = meta_df[['material_id', 'formula', 'knn_distance', 'is_ood', 'pred_proba', 'pred_robust']]
    ood_flags_path = os.path.join(data_dir, 'ood_flags.csv')
    ood_flags_df.to_csv(ood_flags_path, index=False)
    print('OOD flags saved to ' + ood_flags_path)

if __name__ == '__main__':
    main()