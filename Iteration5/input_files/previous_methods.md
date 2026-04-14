1. **Refined Target Definition and Data Integration**:
   - Construct a composite "Mechanical Viability" score ($V_{score}$) for the 90 materials with elastic data: $V_{score} = \text{percentile}(G_{vrh}) \times \exp(-\alpha \cdot \text{energy\_above\_hull})$, where $\alpha$ is a decay constant calibrated to ensure thermodynamic stability significantly penalizes the score.
   - Define the binary target `is_viable` as $V_{score} > \text{median}(V_{score})$ of the stable population.
   - Exclude candidates with `energy_above_hull` > 0.1 eV/atom from the "viable" classification to ensure thermodynamic accessibility.

2. **Handling Missing Data and Selection Bias**:
   - Perform a Mann-Whitney U test to compare `energy_above_hull` and `volume_per_atom` distributions between the 16 missing-DOS samples and the 186 complete samples to assess if missingness is non-random.
   - Implement a "missing-indicator" feature (`is_dos_missing`) and compare model performance against a version using imputed values (using the 5th percentile of observed data to avoid extreme outliers).
   - Retain the approach that yields higher cross-validation stability and explicitly report the feature importance of `is_dos_missing`.

3. **Feature Selection and Collinearity Mitigation**:
   - Calculate the Variance Inflation Factor (VIF) for all continuous features.
   - Perform a baseline model comparison: train a model using only elemental/electronic features vs. one including structural features (`volume_per_atom`, `c_a_ratio`) to determine if structural features add predictive value or merely encode phase labels.
   - Use Recursive Feature Elimination (RFE) with a Random Forest estimator to prune the feature set to a parsimonious subset.

4. **Model Training and Validation**:
   - Train a Random Forest classifier using the refined feature set and the `is_viable` target.
   - Implement Leave-One-Group-Out (LOGO) cross-validation, grouping by "Periodic Table Group" (4, 5, 6, etc.) to ensure the test sets are sufficiently large and diverse.
   - Apply SMOTE to the training folds to address class imbalance.

5. **Evaluation of Physical Drivers**:
   - Conduct a SHAP interaction analysis to investigate the relationship between `d_band_filling` and `dos_at_fermi`.
   - Correlate `M_soc_proxy` with `magnetic_ordering` to determine if the proxy captures electronic physics or magnetic state information.
   - Frame the SHAP analysis of `M_soc_proxy` as a correlation with "heavy-element effects" rather than direct SOC causality.

6. **OOD Detection and Stability Filtering**:
   - Calculate the Mahalanobis distance of each metastable candidate to the centroid of the stable population using only physically relevant features (`d_band_filling`, `en_difference`).
   - Flag candidates with a distance in the top 10th percentile as "chemically exotic" and exclude them from final prioritization.
   - Generate a Pareto front visualization: `energy_above_hull` (x-axis) vs. `predicted_viability_probability` (y-axis), annotated by `phase` to identify the "synthesis window."

7. **Candidate Prioritization and Robustness Check**:
   - Rank the 112 metastable candidates based on their `is_viable` probability score.
   - Apply the final filter: probability > 0.70 and `energy_above_hull` < 0.05 eV/atom.
   - Perform a "leave-one-metal-out" sensitivity check on these high-priority candidates to ensure their high scores are not artifacts of the training data distribution.
   - Categorize final candidates by `phase` and `magnetic_ordering` for experimental validation.