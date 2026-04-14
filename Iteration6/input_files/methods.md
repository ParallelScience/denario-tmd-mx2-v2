1. **Data Refinement and Feature Engineering**:
   - Calculate Pugh’s ratio ($G_{vrh} / K_{vrh}$) as the primary target variable.
   - Incorporate `crystal_system` as a categorical covariate to account for baseline symmetry-dependent variations in $G/K$.
   - Flag high-risk candidates with `energy_above_hull` > 0.05 eV/atom.
   - Create a binary indicator `is_dos_missing` for the 16 samples lacking `dos_at_fermi` and use mean imputation (derived strictly from training folds) for the missing values.
   - Include `c_a_ratio` as a feature to capture the coupling between lattice anisotropy and mechanical stability.

2. **Hierarchical Model Architecture**:
   - Implement a shallow neural network (2-3 dense layers) with a shared feature extraction backbone and a "Group Embedding" layer to represent transition metal groups (4-7 vs 8-10).
   - Utilize a multi-task learning framework: the primary head performs regression on Pugh’s ratio (using the 90-sample subset), while a secondary classification head predicts `is_stable` (using the full 202-sample set) to regularize the backbone.
   - Apply Dropout and L2 regularization to the shared backbone to prevent overfitting on the small dataset.

3. **Handling Missing Data and Uncertainty**:
   - Perform sensitivity analysis on the 16 missing `dos_at_fermi` samples by comparing model performance with and without the `is_dos_missing` indicator.
   - Use Deep Ensembles (training multiple models with different initializations) to estimate epistemic uncertainty for the Pugh’s ratio predictions, ensuring the "Confidence Interval" is robust.

4. **Group-Aware Cross-Validation and Interpretability**:
   - Execute Leave-One-Group-Out (LOGO) cross-validation, grouping by transition metal.
   - If neural network performance is insufficient, switch to a Random Forest Regressor.
   - Apply SHAP (SHapley Additive exPlanations) values to both models to visualize non-linear interactions between `d_band_filling`, `en_difference`, and `dos_at_fermi`, ensuring the model captures physically meaningful trends.

5. **Mechanical Viability Mapping**:
   - Define the composite viability score $S = \text{Predicted}(G/K) \times \text{Uncertainty}^{-1}$.
   - Generate a 2D probability map plotting predicted Pugh’s ratio vs. `energy_above_hull`, highlighting the 0.05 eV/atom stability threshold.
   - Filter out candidates with `dos_at_fermi` values that are extreme outliers (> 3 standard deviations from the mean) to avoid metallic artifacts.

6. **Candidate Prioritization**:
   - Rank the 112 metastable candidates based on the composite viability score $S$.
   - Apply the "Stability-Robustness" filter (0.05 eV/atom) and prioritize candidates where `theoretical = True`.

7. **DFT-Validation of Top Candidates**:
   - Select the top 3 candidates for validation.
   - Perform PBE+U calculations (applying U selectively to metal d-orbitals using Materials Project standard values) to verify the elastic tensor.
   - Include a check for dynamical stability by ensuring the structure remains a local minimum on the potential energy surface during relaxation and checking for spontaneous symmetry breaking.

8. **Final Robustness Check**:
   - Conduct a "leave-one-metal-out" sensitivity check on the top 10 candidates to ensure predictions are not driven by a single metal-chalcogen combination.
   - Validate that the final selection aligns with established solid-state chemistry principles regarding the correlation between $G/K$ and `d_band_filling`.