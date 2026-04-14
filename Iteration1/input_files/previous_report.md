

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
        