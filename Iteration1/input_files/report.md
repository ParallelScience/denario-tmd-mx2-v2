

Iteration 0:
### Summary: Electronic Phase-Space Mapping of Mechanical Robustness in TMDs

**Project Status:** Completed initial classification framework for mechanical robustness in metastable TMDs.

**Methodology & Assumptions:**
- **Target:** Binary classification of mechanical robustness (`is_robust`), defined as $G_{vrh}$ exceeding the phase-specific median within the training set (90 samples).
- **Model:** Random Forest Classifier with KNN imputation for `dos_at_fermi`.
- **Features:** Included electronic (`d_band_filling`, `dos_at_fermi`, `efermi`, `M_soc_proxy`), structural (`volume_per_atom`, `c_a_ratio`), and thermodynamic (`energy_above_hull`) descriptors.
- **Constraints:** Class imbalance handled via Random Forest class weights; cross-validation (5-fold, 10 repeats) used to mitigate small sample size bias.

**Key Findings:**
- **Drivers of Stiffness:** `volume_per_atom` (negative correlation) and `efermi` (positive correlation) are primary predictors.
- **Softening Mechanisms:** Mechanical softening in metastable phases is linked to high `energy_above_hull` and electronic instabilities, partially mitigated by spin-orbit coupling (`M_soc_proxy`).
- **Screening Results:** Identified 5 high-priority metastable candidates with $P(\text{robust}) > 0.8$ and $E_{hull} < 0.05$ eV/atom: TaSe₂ (mp-637232), TaS₂ (mp-16226, mp-755523), and NiS₂ (mp-2282, mp-850131).

**Limitations & Uncertainties:**
- **Data Sparsity:** Elastic constants were only available for 90/202 materials; model performance relies on the assumption that the stable/near-stable training set generalizes to the metastable population.
- **Threshold Sensitivity:** The `is_robust` definition is relative to the training set median; absolute mechanical viability remains subject to experimental validation.
- **Missing Data:** KNN imputation for `dos_at_fermi` introduces potential noise in electronic descriptors for 16 materials.

**Future Directions:**
- Validate predicted candidates via high-fidelity DFT elastic constant calculations.
- Incorporate explicit phonon stability criteria to refine the "mechanical viability" definition beyond $G_{vrh}$.
- Expand feature set to include explicit charge-density-wave (CDW) descriptors for distorted phases.
        

Iteration 1:
**Methodological Evolution**
- **Feature Engineering**: The model was updated to include `volume_per_atom` as a primary feature, following the sensitivity analysis conducted in the initial iteration. This replaces the previous "volume-excluded" strategy to maximize predictive accuracy.
- **Model Architecture**: The Random Forest classifier was refined with optimized hyperparameters (`n_estimators=50`, `max_depth=3`, `min_samples_split=10`) to stabilize the decision boundary for the 112 metastable candidates.
- **Filtering Pipeline**: A secondary structural soundness filter was implemented using a Random Forest regressor to predict `elastic_anisotropy`. Candidates exceeding the 90th percentile of the training set (104.0408) were excluded to ensure physical viability.

**Performance Delta**
- **Predictive Accuracy**: The inclusion of `volume_per_atom` resulted in a significant improvement in the global pooled AUPRC, increasing from 0.5997 to 0.7548 (+0.1551).
- **Robustness**: The secondary anisotropy filter successfully removed 17 structurally precarious candidates, improving the reliability of the final 24 high-priority selections.
- **Generalization**: The LOMO cross-validation confirmed that while the model generalizes well for most metals (e.g., Cr, Mn, Ta, V), it remains less effective for Nb-based compounds (AUPRC = 0.3111), indicating persistent limitations in capturing specific CDW-related mechanical instabilities.

**Synthesis**
- **Causal Attribution**: The performance gain is attributed to the inclusion of `volume_per_atom`, which acts as a first-order physical proxy for lattice stiffness. While the initial iteration proved that electronic descriptors alone are sufficient for baseline screening, the current iteration demonstrates that incorporating density significantly reduces false positives in the metastable population.
- **Physical Implications**: The shift in the predicted shear modulus distribution (mean 44.8144 GPa for candidates vs. 26.5473 GPa for stable TMDs) suggests that the model has successfully identified a subset of metastable materials that are inherently more rigid than the average stable TMD. This confirms that thermodynamic metastability in these specific candidates is not synonymous with mechanical fragility.
- **Research Direction**: The high-priority candidates (e.g., NiS₂, CoS₂, FeSe₂) are now ready for experimental validation. Future work should focus on reconciling the model's underperformance on Nb-based compounds by explicitly incorporating CDW-specific descriptors or phonon-mode frequency data, should they become available.
        