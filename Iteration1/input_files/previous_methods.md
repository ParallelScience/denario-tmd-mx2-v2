1. **Data Preprocessing and Feature Engineering**: 
   - Impute missing `dos_at_fermi` values using a KNN regressor trained only on the training folds during cross-validation. Include `band_gap`, `M_val`, `X_period`, and `phase` as features to capture the physical constraint that semiconducting materials have zero DOS at the Fermi level.
   - Encode categorical variables: apply one-hot encoding to `magnetic_ordering` and `phase`, and ordinal encoding to `crystal_system`.
   - Incorporate structural descriptors `c_a_ratio`, `volume_per_atom`, and `energy_above_hull` as features to account for lattice packing and thermodynamic stability.
   - Standardize all continuous features to have zero mean and unit variance.

2. **Target Definition for Mechanical Stiffness**:
   - Define the target `is_robust` based on the shear modulus ($G_{vrh}$). To account for structural variance, calculate the median $G_{vrh}$ per `crystal_system` or `phase` within the training set to define the threshold for "stiffness."
   - Assign `1` to materials where $G_{vrh}$ is above the calculated median and `0` otherwise. Ensure this thresholding is performed strictly within cross-validation folds to prevent label leakage.

3. **Exploratory Correlation Analysis**:
   - Perform Spearman rank correlation analysis between `is_robust` and the primary electronic/structural descriptors.
   - Visualize the feature space using pair-plots colored by `is_robust` to identify non-linear decision boundaries and clusters of mechanical stiffness.

4. **Handling Class Imbalance**:
   - Assess the distribution of the `is_robust` target in the 90-sample training set.
   - Apply class weight balancing within the Random Forest classifier to mitigate bias toward the majority class, avoiding synthetic data generation (SMOTE) to preserve the physical integrity of the small dataset.

5. **Model Selection and Training**:
   - Implement a Random Forest Classifier to capture non-linear interactions.
   - Use Repeated Stratified K-Fold Cross-Validation (e.g., 5-fold, 10 repeats) to ensure stable performance estimates and handle the small sample size.
   - Optimize hyperparameters (`n_estimators`, `max_depth`, `min_samples_split`) using AUPRC as the primary metric.

6. **Feature Importance and Sensitivity Analysis**:
   - Calculate Permutation Feature Importance to rank the influence of electronic and structural features on mechanical stiffness.
   - Generate Partial Dependence Plots (PDPs) for the top features to visualize how electronic filling and structural parameters promote or suppress mechanical stiffness.

7. **Inference on Metastable Candidates**:
   - Apply the trained Random Forest model to the 112 metastable materials.
   - Generate a probability score for each material representing its likelihood of being mechanically robust.

8. **Validation and Prioritization**:
   - Perform a sanity check by comparing the distribution of `energy_above_hull` between predicted "robust" and "non-robust" metastable candidates.
   - Filter candidates using a dual criterion: high model-predicted probability ($>0.8$) and a specific thermodynamic stability threshold (e.g., `energy_above_hull` < 0.05 eV/atom) to identify the most promising candidates for experimental synthesis.