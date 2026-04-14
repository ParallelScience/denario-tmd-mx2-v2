1. **Data Preprocessing and Feature Selection**:
   - Perform conditional imputation for `dos_at_fermi`: for all materials where `band_gap > 0.1` eV, explicitly set `dos_at_fermi = 0`. For the remaining metallic samples, use a KNN regressor (incorporating `band_gap` as a feature) to impute missing values, ensuring imputation is restricted to training folds.
   - Exclude `volume_per_atom` and `energy_above_hull` from the primary feature set to focus on electronic and compositional descriptors (`d_band_filling`, `M_soc_proxy`, `en_difference`, `M_atomic_radius`, `X_atomic_radius`).
   - Encode categorical variables: one-hot encode `magnetic_ordering` and `phase`, and ordinal encode `crystal_system`. Standardize all continuous features to zero mean and unit variance.

2. **Global Target Definition**:
   - Calculate the median of `G_vrh` across the 90-sample elastic dataset.
   - Define the binary target `is_robust` using this median as the threshold: assign `1` to materials where `G_vrh` is greater than the median, and `0` otherwise, ensuring a balanced 50/50 split for the classification task.

3. **Leave-One-Metal-Out (LOMO) Cross-Validation**:
   - Implement a LOMO validation strategy where the model is trained on all materials except those containing a specific transition metal.
   - Repeat this for all 13 transition metals to evaluate the model's ability to generalize to unseen chemical spaces.

4. **Model Training and Hyperparameter Optimization**:
   - Train a Random Forest Classifier using the LOMO folds.
   - Optimize hyperparameters (`n_estimators`, `max_depth`, `min_samples_split`) using AUPRC as the primary metric.
   - Apply class weight balancing to ensure the model effectively learns from both classes.

5. **Feature Importance and Physical Interpretation**:
   - Calculate Permutation Feature Importance across the LOMO folds to assess the model's reliance on fundamental electronic descriptors versus structural proxies.
   - Generate Partial Dependence Plots (PDPs) for the top-performing features to visualize the relationship between electronic filling and the probability of mechanical robustness.

6. **Mechanical Stability Filtering**:
   - For the 112 metastable candidates, calculate the predicted probability of being `is_robust`.
   - Apply a secondary filter for structural soundness: exclude candidates with `elastic_anisotropy` exceeding the 90th percentile of the training set to ensure predicted "stiff" materials are not prone to extreme lattice distortions.

7. **Candidate Prioritization**:
   - Rank the metastable candidates based on high model-predicted probability ($>0.75$) and acceptable `elastic_anisotropy`.
   - Compare the distribution of these high-priority candidates against the known stable population to verify physical plausibility.

8. **Sensitivity Analysis**:
   - Re-run the model including `volume_per_atom` as a feature to quantify the "information gain" provided by density. Compare the AUPRC of this model against the density-excluded model to confirm the predictive sufficiency of the electronic/compositional descriptors.