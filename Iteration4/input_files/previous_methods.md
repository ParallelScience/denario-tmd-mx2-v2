1. **Data Preprocessing and Feature Engineering**:
   - Perform KNN-imputation for the 16 missing `dos_at_fermi` values using the entire dataset. Include a binary indicator feature `is_dos_missing` to allow the model to learn the reliability of these imputed values.
   - Compare the distribution of imputed values against the global distribution of the 186 known samples to ensure physical plausibility.
   - Apply one-hot encoding to categorical variables (`phase`, `crystal_system`, `magnetic_ordering`).
   - Standardize all continuous features using robust scaling (median/IQR) to mitigate the influence of outliers.

2. **Target Definition and Sensitivity Analysis**:
   - Define the `is_robust` target as a binary classification based on the median $G_{vrh}$ of the stable population.
   - Conduct a sensitivity analysis by varying the robustness threshold between the 60th and 80th percentiles of the stable population's $G_{vrh}$.
   - Define the "high-confidence" set as the intersection of candidates classified as robust across all tested thresholds.
   - Document the stability of feature importance (via SHAP) across these thresholds to identify if the drivers of mechanical robustness shift with the definition.

3. **Model Architecture and Training**:
   - Train a Random Forest classifier to predict `is_robust` using the full 202-sample dataset.
   - Use Leave-One-Cluster-Out (LOCO) cross-validation, grouping by transition metal to ensure generalization across chemical families.
   - Apply class weight balancing to the `is_robust` target to account for the imbalance between stable and metastable phases.

4. **Feature Ablation and Hypothesis Testing**:
   - Perform a feature ablation study by training the model with and without `M_soc_proxy` and `dos_at_fermi`.
   - Compare the AUPRC of these models to quantify the contribution of spin-orbit coupling and electronic density of states to mechanical robustness.
   - If the performance drop upon removing `M_soc_proxy` is negligible, reject the hypothesis that SOC is a primary driver of mechanical softening.

5. **Out-of-Distribution (OOD) Detection**:
   - Calculate the KNN distance of each metastable candidate to its 5 nearest neighbors in the stable training set using the standardized feature space.
   - Flag candidates with a distance exceeding the 95th percentile of the stable population's internal distance distribution as "chemically exotic" or OOD.
   - Generate a 2D projection (e.g., PCA or t-SNE) of the feature space, coloring points by "Stable," "Metastable-Robust," and "Metastable-OOD" to visualize the distribution.

6. **Model Interpretation**:
   - Utilize SHAP values to quantify the contribution of `d_band_filling`, `en_difference`, and `dos_at_fermi` to classification decisions.
   - Generate summary plots to visualize non-linear interactions between electronic descriptors and mechanical robustness.
   - Specifically analyze the interaction effect between `d_band_filling` and `dos_at_fermi` for Ni-based TMDs to determine if robust candidates correlate with a local minimum in the DOS at the Fermi level.

7. **Candidate Prioritization**:
   - Rank the 112 metastable candidates by their predicted probability of being `is_robust` ($>0.75$).
   - Filter the final list by excluding candidates flagged as OOD by the KNN distance metric.
   - Provide a final prioritized list of candidates, categorized by their proximity to known stable phases and their electronic descriptor profiles.