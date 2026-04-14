# filename: codebase/step_7.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import joblib
import json

def main():
    data_dir = "data"
    processed_data_path = os.path.join(data_dir, "processed_tmd_data.csv")
    targets_path = os.path.join(data_dir, "target_labels.csv")
    features_path = os.path.join(data_dir, "training_features.json")
    model_path = os.path.join(data_dir, "rf_model.joblib")
    ood_flags_path = os.path.join(data_dir, "ood_flags.csv")
    
    df_processed = pd.read_csv(processed_data_path)
    df_targets = pd.read_csv(targets_path)
    ood_flags_df = pd.read_csv(ood_flags_path)
    
    with open(features_path, "r") as f:
        features = json.load(f)
        
    rf_model = joblib.load(model_path)
    
    train_mids = set(df_targets['material_id'])
    meta_df = df_processed[~df_processed['material_id'].isin(train_mids)].copy()
    
    X_meta = meta_df[features]
    meta_df['pred_proba'] = rf_model.predict_proba(X_meta)[:, 1]
    
    meta_df = pd.merge(meta_df, ood_flags_df[['material_id', 'is_ood', 'knn_distance']], on='material_id', how='left')
    
    filtered_df = meta_df[(meta_df['is_ood'] == False) & (meta_df['pred_proba'] > 0.75)].copy()
    
    ranked_df = filtered_df.sort_values(by='pred_proba', ascending=False)
    
    print("Total metastable candidates: " + str(len(meta_df)))
    print("Candidates after filtering OOD and pred_proba > 0.75: " + str(len(ranked_df)))
    print("\nTop Prioritized Candidates for Experimental Synthesis:")
    print("Rank  | Material ID     | Formula    | Pred Proba   | KNN Dist   | d_band_filling | dos_at_fermi")
    print("-" * 95)
    
    rank = 1
    for _, row in ranked_df.iterrows():
        rank_str = str(rank).ljust(5)
        m_id = str(row['material_id']).ljust(15)
        formula = str(row['formula']).ljust(10)
        proba = str(round(row['pred_proba'], 4)).ljust(12)
        knn_dist = str(round(row['knn_distance'], 4)).ljust(10)
        d_band = str(round(row['d_band_filling'], 4)).ljust(14)
        dos = str(round(row['dos_at_fermi'], 4)).ljust(12)
        print(rank_str + " | " + m_id + " | " + formula + " | " + proba + " | " + knn_dist + " | " + d_band + " | " + dos)
        rank += 1
        
    output_cols = ['material_id', 'formula', 'pred_proba', 'knn_distance', 'is_ood'] + features
    output_df = ranked_df[output_cols]
    output_path = os.path.join(data_dir, "prioritized_candidates.csv")
    output_df.to_csv(output_path, index=False)
    print("\nPrioritized candidates saved to " + output_path)

if __name__ == '__main__':
    main()