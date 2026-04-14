# filename: codebase/step_3.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

if __name__ == '__main__':
    data_dir = 'data'
    data_path = os.path.join(data_dir, 'processed_tmd_data.csv')
    df = pd.read_csv(data_path)
    encoded_cols = [col for col in df.columns if col.startswith('crystal_system_') or col.startswith('phase_') or col.startswith('magnetic_ordering_')]
    base_features = ['d_band_filling', 'en_difference', 'dos_at_fermi', 'c_a_ratio', 'M_soc_proxy', 'M_group'] + encoded_cols
    features_with_missing = base_features + ['is_dos_missing']
    if df['c_a_ratio'].isna().sum() > 0:
        df['c_a_ratio'] = df['c_a_ratio'].fillna(df['c_a_ratio'].mean())
    df_reg = df.dropna(subset=['pugh_ratio']).copy()
    groups = df_reg['metal'].values
    logo = LeaveOneGroupOut()
    def evaluate_features(features_list):
        all_y_true = []
        all_y_pred = []
        for train_idx, test_idx in logo.split(df_reg, groups=groups):
            train_df = df_reg.iloc[train_idx].copy()
            test_df = df_reg.iloc[test_idx].copy()
            dos_mean = train_df['dos_at_fermi'].mean()
            train_df['dos_at_fermi'] = train_df['dos_at_fermi'].fillna(dos_mean)
            test_df['dos_at_fermi'] = test_df['dos_at_fermi'].fillna(dos_mean)
            X_train = train_df[features_list]
            y_train = train_df['pugh_ratio']
            X_test = test_df[features_list]
            y_test = test_df['pugh_ratio']
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)
            all_y_true.extend(y_test)
            all_y_pred.extend(y_pred)
        mae = mean_absolute_error(all_y_true, all_y_pred)
        rmse = np.sqrt(mean_squared_error(all_y_true, all_y_pred))
        r2 = r2_score(all_y_true, all_y_pred)
        return mae, rmse, r2
    print('--- Sensitivity Analysis on Missing Data ---')
    mae_without, rmse_without, r2_without = evaluate_features(base_features)
    print('Performance WITHOUT is_dos_missing:')
    print('  MAE:  ' + str(round(mae_without, 4)))
    print('  RMSE: ' + str(round(rmse_without, 4)))
    print('  R2:   ' + str(round(r2_without, 4)))
    mae_with, rmse_with, r2_with = evaluate_features(features_with_missing)
    print('\nPerformance WITH is_dos_missing:')
    print('  MAE:  ' + str(round(mae_with, 4)))
    print('  RMSE: ' + str(round(rmse_with, 4)))
    print('  R2:   ' + str(round(r2_with, 4)))
    print('\n--- Training Ensemble for Epistemic Uncertainty ---')
    n_models = 10
    ensemble_models = []
    dos_mean_full = df['dos_at_fermi'].mean()
    df_imputed = df.copy()
    df_imputed['dos_at_fermi'] = df_imputed['dos_at_fermi'].fillna(dos_mean_full)
    df_reg_imputed = df_imputed.dropna(subset=['pugh_ratio']).copy()
    X_train_full = df_reg_imputed[features_with_missing]
    y_train_full = df_reg_imputed['pugh_ratio']
    for i in range(n_models):
        rf = RandomForestRegressor(n_estimators=100, random_state=i*42)
        rf.fit(X_train_full, y_train_full)
        ensemble_models.append(rf)
    print('Trained ' + str(n_models) + ' RandomForest models with different seeds.')
    X_all = df_imputed[features_with_missing]
    predictions = np.zeros((len(df_imputed), n_models))
    for i, model in enumerate(ensemble_models):
        predictions[:, i] = model.predict(X_all)
    mean_preds = np.mean(predictions, axis=1)
    std_preds = np.std(predictions, axis=1)
    df_imputed['mean_pugh_prediction'] = mean_preds
    df_imputed['epistemic_uncertainty'] = std_preds
    output_cols = ['material_id', 'formula', 'metal', 'chalcogen', 'phase', 'energy_above_hull', 'is_stable', 'mean_pugh_prediction', 'epistemic_uncertainty', 'dos_at_fermi', 'is_dos_missing']
    additional_cols = ['d_band_filling', 'en_difference', 'theoretical']
    for col in additional_cols:
        if col in df_imputed.columns and col not in output_cols:
            output_cols.append(col)
    uncertainty_df = df_imputed[output_cols]
    output_path = os.path.join(data_dir, 'uncertainty_metrics.csv')
    uncertainty_df.to_csv(output_path, index=False)
    ensemble_path = os.path.join(data_dir, 'ensemble_models.joblib')
    joblib.dump({'models': ensemble_models, 'features': features_with_missing}, ensemble_path)
    print('\nUncertainty metrics saved to: ' + output_path)
    print('Ensemble models saved to: ' + ensemble_path)
    print('\nSummary of Epistemic Uncertainty:')
    print(uncertainty_df['epistemic_uncertainty'].describe().to_string())
    print('\nTop 5 candidates with highest uncertainty:')
    print(uncertainty_df.sort_values('epistemic_uncertainty', ascending=False).head(5).to_string(index=False))
    print('\nTop 5 candidates with lowest uncertainty:')
    print(uncertainty_df.sort_values('epistemic_uncertainty', ascending=True).head(5).to_string(index=False))