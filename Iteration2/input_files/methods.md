1. **Data Preprocessing and Imputation Strategy**:
   - Perform KNN-imputation for the 16 missing `dos_at_fermi` values. Calculate the standard deviation of these imputed values across K-neighbors to quantify imputation uncertainty; store this as a new feature `imputation_uncertainty`.
   - Add a binary indicator feature `is_dos_missing` to allow the model to learn if the absence of a DOS calculation is a predictor of instability.
   - Standardize all continuous features using robust scaling. Encode categorical variables (`phase`, `crystal_system`) using target encoding.
   - Perform a Kolmogorov-Smirnov test on `d_band_filling` and `M_en` to assess dataset shift between the 90 materials with elastic data and the 112 without.

2. **Multi-Task Target Definition and Loss Weighting**:
   - Define `is_robust` (binary) based on the 70th percentile of `G_vrh` within the stable population.
   - Define `is_stable_structure` (regression) based on `elastic_anisotropy`.
   - Implement a multi-task learning framework where the loss function is $\mathcal{L}_{total} = \alpha \cdot \mathcal{L}_{class} + (1-\alpha) \cdot \mathcal{L}_{reg}$. Use a masked loss for the regression component so that it only updates weights based on the 90 samples where ground truth exists, while the classification component utilizes the full dataset. Tune $\alpha$ to prevent regression from dominating.

3. **Refined LOCO Cross-Validation**:
   - Group transition metals into clusters based on periodic table groups or d-electron count ranges to ensure balanced sample sizes across folds.
   - Perform Leave-One-Cluster-Out (LOCO) cross-validation. If cluster sizes remain highly uneven, use stratified grouping to ensure each fold contains a representative mix of stable and metastable phases.

4. **Baseline vs. Full Model Comparison**:
   - Train a baseline model using static descriptors (`X_atomic_radius`, `crystal_system`, `M_group`).
   - Train the full model using the complete feature set including electronic descriptors (`d_band_filling`, `dos_at_fermi`, `M_soc_proxy`).
   - Compare AUPRC performance to quantify the gain from electronic phase-space dynamics.

5. **Physical Correlation and OOD Analysis**:
   - Calculate the Spearman rank correlation between `M_soc_proxy` and `G_vrh` for the stable population, including p-value calculations to ensure statistical significance.
   - Calculate the Mahalanobis distance of each metastable candidate from the centroid of the stable training population in the feature space to identify and flag "out-of-distribution" (OOD) candidates.

6. **Model Training and Optimization**:
   - Train the Random Forest ensemble using the LOCO folds.
   - Apply class weight balancing to the `is_robust` target to account for the threshold-induced imbalance.
   - Perform hyperparameter optimization focused on maximizing AUPRC for the classification task while maintaining regression stability.

7. **Feature Interpretation and Visualization**:
   - Use SHAP values to assess non-linear interactions between features.
   - Generate 2D Accumulated Local Effects (ALE) plots for `d_band_filling` and `en_difference` to visualize the stability boundary.
   - Overlay scatter plots of the 46 stable materials onto the ALE plots to verify if the model's high-probability regions align with known islands of stability.

8. **Candidate Prioritization**:
   - Rank the 112 metastable candidates by their predicted probability of being `is_robust` ($>0.75$) and predicted `elastic_anisotropy` ($<90$th percentile).
   - Filter the final list by excluding candidates with high `imputation_uncertainty` or high Mahalanobis distance (OOD).
   - Provide the final prioritized list with confidence flags based on data quality and OOD status.