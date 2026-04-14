1. **Data Preprocessing and Imputation**:
   - Split the dataset into "Elasticity-Known" (90 samples) and "Elasticity-Unknown" (112 samples).
   - Perform iterative imputation for missing `dos_at_fermi` values using `IterativeImputer` with a BayesianRidge estimator. Crucially, fit the imputer only on the training folds during cross-validation to prevent data leakage.
   - Standardize continuous features (e.g., `d_band_filling`, `en_difference`, `dos_at_fermi`, `M_soc_proxy`, `c_a_ratio`) to zero mean and unit variance.
   - Z-score normalize `G_vrh` to a [0, 1] range to ensure the regression head does not dominate the multi-task loss.

2. **Multi-Task Model Architecture**:
   - Construct a neural network with a shared feature extraction backbone (dense layers with ReLU activation).
   - Define two output heads: a regression head for normalized `G_vrh` and a classification head for `is_stable`.
   - Implement homoscedastic uncertainty weighting for the loss function $L = \frac{1}{2\sigma_1^2} \text{MSE}(G_{vrh}) + \frac{1}{2\sigma_2^2} \text{BCE}(is\_stable) + \log(\sigma_1\sigma_2)$ to dynamically balance the regression and classification tasks.

3. **Feature Engineering and Regularization**:
   - Include `d_band_filling`, `en_difference`, `bond_radius_sum`, and `c_a_ratio` as primary physical descriptors.
   - Apply ElasticNet regularization to the shared backbone, but apply a custom penalty mask to ensure `d_band_filling` and `en_difference` are protected from being dropped.
   - One-hot encode `magnetic_ordering` and `crystal_system` to capture categorical structural influences.

4. **Model Training and Validation**:
   - Train the model using the 90-sample elasticity dataset for the regression head and the full 202-sample dataset for the stability head.
   - Implement Leave-One-Group-Out (LOGO) cross-validation, grouping by "Periodic Table Group," ensuring stratified splits within folds to maintain chemical diversity.
   - Monitor the correlation between predicted `G_vrh` and `energy_above_hull` to ensure the model captures the physical "softening" trend in metastable phases.

5. **Sensitivity and Uncertainty Analysis**:
   - Perform a Monte Carlo sensitivity check by running multiple forward passes with jittered input features to quantify the model's confidence in the predicted `G_vrh`.
   - For candidates with high `d_band_filling` and FM/FiM magnetic ordering, assign a "High-Correlation Sensitivity" flag, noting that PBE-level predictions for these late transition metals may overestimate mechanical robustness.

6. **Mechanical Viability Mapping**:
   - Define "Mechanical Viability" using a Z-score threshold: $G_{vrh} > \mu_{stable} + \sigma_{stable}$ (based on the stable population distribution).
   - Calculate "Specific Shear Modulus" ($G_{vrh} / \rho$) as a secondary metric to ensure candidates possess high intrinsic bond-stiffness rather than just high density.
   - Generate a 2D probability map plotting predicted $G_{vrh}$ vs. `energy_above_hull`, color-coded by the model's uncertainty.

7. **Candidate Prioritization**:
   - Rank the 112 metastable candidates based on the viability score.
   - Filter for candidates with `energy_above_hull` < 0.05 eV/atom and high predicted $G_{vrh}$.
   - Cross-reference with the `theoretical` flag to prioritize previously unsynthesized materials.

8. **Final Robustness Check**:
   - Perform a "leave-one-metal-out" sensitivity check on the top 10 prioritized candidates to ensure the predicted mechanical viability is not driven by a single metal-chalcogen combination.
   - Validate that the final selection follows the expected physical trend of $G_{vrh}$ vs. `d_band_filling` for TMDs.