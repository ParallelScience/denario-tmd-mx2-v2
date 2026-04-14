# filename: codebase/step_2.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

plt.rcParams['text.usetex'] = False

if __name__ == '__main__':
    data_dir = 'data/'
    df = pd.read_csv(os.path.join(data_dir, 'processed_tmd_data.csv'))
    df_elastic = df.dropna(subset=['G_vrh']).copy()
    print('Number of materials with elastic data: ' + str(len(df_elastic)))
    phase_stats = df_elastic.groupby('phase')['G_vrh'].agg(['count', 'median'])
    print('\nElastic data stats per phase:')
    for p, row in phase_stats.iterrows():
        print('Phase ' + str(p) + ' - Count: ' + str(int(row['count'])) + ', Median G_vrh: ' + str(round(row['median'], 2)) + ' GPa')
    phase_medians = phase_stats['median']
    df['phase_median_G_vrh'] = df['phase'].map(phase_medians)
    df['is_robust'] = np.where(df['G_vrh'].notna(), (df['G_vrh'] > df['phase_median_G_vrh']).astype(float), np.nan)
    df_elastic = df.dropna(subset=['G_vrh']).copy()
    print('\nDistribution of is_robust in the elastic subset:')
    print(df_elastic['is_robust'].astype(int).value_counts().to_string())
    continuous_features = ['volume', 'nsites', 'volume_per_atom', 'a', 'b', 'c', 'alpha', 'beta', 'gamma', 'c_a_ratio', 'energy_above_hull', 'log1p_energy_above_hull', 'formation_energy_per_atom', 'band_gap', 'efermi', 'dos_at_fermi', 'total_magnetization', 'M_val', 'M_Z', 'M_en', 'M_ie1', 'M_atomic_radius', 'M_group', 'M_period', 'M_soc_proxy', 'X_en', 'X_ie1', 'X_atomic_radius', 'X_period', 'en_difference', 'bond_radius_sum', 'd_count_m4plus', 'd_band_filling']
    corr_matrix = df_elastic[continuous_features + ['is_robust']].corr(method='spearman')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    corr_csv_path = os.path.join(data_dir, 'spearman_correlation_matrix_' + timestamp + '.csv')
    corr_matrix.to_csv(corr_csv_path)
    print('\nCorrelation matrix saved to ' + corr_csv_path)
    corr_with_target = corr_matrix['is_robust'].drop('is_robust').dropna()
    corr_with_target_sorted = corr_with_target.reindex(corr_with_target.abs().sort_values(ascending=False).index)
    print('\nSpearman correlations with is_robust:')
    print(corr_with_target_sorted.to_string())
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(x=corr_with_target_sorted.values, y=corr_with_target_sorted.index, ax=ax, color='steelblue')
    ax.set_title('Spearman Correlation with is_robust')
    ax.set_xlabel('Spearman Correlation Coefficient')
    ax.set_ylabel('Features')
    fig.tight_layout()
    bar_chart_path = os.path.join(data_dir, 'correlation_bar_chart_1_' + timestamp + '.png')
    fig.savefig(bar_chart_path, dpi=300)
    print('Bar chart saved to ' + bar_chart_path)
    plt.close(fig)
    pairs = [('d_band_filling', 'en_difference'), ('dos_at_fermi', 'volume_per_atom'), ('M_soc_proxy', 'c_a_ratio')]
    df_elastic['is_robust_cat'] = df_elastic['is_robust'].astype(int).astype(str)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, (x_feat, y_feat) in enumerate(pairs):
        sns.scatterplot(data=df_elastic, x=x_feat, y=y_feat, hue='is_robust_cat', palette={'0': 'blue', '1': 'red'}, ax=axes[i], alpha=0.7)
        axes[i].set_title(x_feat + ' vs ' + y_feat)
        axes[i].set_xlabel(x_feat + ' (standardized)')
        axes[i].set_ylabel(y_feat + ' (standardized)')
    fig.tight_layout()
    pair_plot_path = os.path.join(data_dir, 'feature_pair_plots_2_' + timestamp + '.png')
    fig.savefig(pair_plot_path, dpi=300)
    print('Pair plots saved to ' + pair_plot_path)
    plt.close(fig)
    df_target_path = os.path.join(data_dir, 'processed_tmd_data_with_target.csv')
    df.to_csv(df_target_path, index=False)
    print('\nFull dataset with is_robust target saved to ' + df_target_path)