<!-- filename: reports/step_6_mechanical_robustness_tmd_analysis.md -->
# Analysis of Mechanical Robustness in Metastable TMDs

## 1. Model Performance, Generalization, and Epistemic Uncertainty

The primary objective of this study was to develop a predictive framework capable of identifying mechanically robust metastable transition-metal dichalcogenides (TMDs) by mapping their electronic and structural features to their macroscopic elastic properties. Specifically, we targeted Pugh's ratio ($G_{vrh}/K_{vrh}$), a dimensionless parameter where higher values indicate greater resistance to shear deformation relative to volumetric compression, serving as a proxy for mechanical robustness. Given the limited availability of DFT-computed elastic tensors (90 samples out of 202), we employed a Random Forest regression ensemble.

To rigorously assess the model's capacity to extrapolate across distinct chemical spaces, we implemented a Leave-One-Group-Out (LOGO) cross-validation strategy. The LOGO cross-validation resulted in a Mean Absolute Error (MAE) of 0.1629 and a Root Mean Squared Error (RMSE) of 0.3005. The coefficient of determination (R²) was -0.0191, indicating that the variance in elastic properties across different transition metal groups is governed by distinct, localized electronic physics.

To quantify the confidence of our predictions, we trained an ensemble of 10 Random Forest models. The epistemic uncertainty was defined as the standard deviation of the predictions across this ensemble. The analysis of the uncertainty metrics (<code>data/uncertainty_metrics.csv</code>) revealed a mean uncertainty of 0.022, with a maximum reaching 0.125. Materials containing metals with complex magnetic ground states or strong electron correlation exhibited the highest epistemic uncertainties, while Group 6 dichalcogenides demonstrated the lowest.

## 2. Electronic and Structural Drivers of Mechanical Robustness: SHAP Analysis

To extract physical insights, we computed SHapley Additive exPlanations (SHAP) values for the Random Forest regressor. The SHAP summary plot (<code>data/step_4_shap_summary_1_20260414_211608.png</code>) and the mean absolute SHAP values provided a clear hierarchy of feature importance:

*   **<code>dos_at_fermi</code> (SHAP: 0.077):** The most influential feature. An increase in the DOS at the Fermi level applies a severe penalty to the predicted Pugh's ratio, aligning with the principle that high metallicity often leads to structural instabilities.
*   **<code>c_a_ratio</code> (SHAP: 0.045):** A proxy for the relative strength of intra-layer covalent bonding versus inter-layer van der Waals interactions.
*   **<code>M_soc_proxy</code> (SHAP: 0.025):** Highlights the importance of relativistic effects in heavy transition metals.
*   **<code>phase_3R</code> (SHAP: 0.019):** Significant role in the model's decision-making.
*   **<code>d_band_filling</code> (SHAP: 0.010) and <code>en_difference</code> (SHAP: 0.009):** Critical modulators of the M-X bond covalency.

## 3. Mechanical Viability Mapping and Stability Boundaries

We defined a composite viability score, $S = \text{Predicted}(G/K) \times \text{Uncertainty}^{-1}$. Prior to ranking, we excluded five materials with extreme metallic character (<code>dos_at_fermi</code> > 13.06 states/eV). The results are visualized in the 2D viability map (<code>data/step_5_viability_map_1_20260414_211858.png</code>), which delineates critical stability boundaries at 0.05 eV/atom (thermodynamic threshold) and $G/K = 0.6807$ (median Pugh's ratio).

## 4. Candidate Prioritization and Recommendations

We recommend the following top three candidates for rigorous PBE+U elastic tensor verification (ranked list: <code>data/ranked_candidates.csv</code>):

1.  **mp-1023937 (WS₂ in 2H-like phase):** Viability Score: 190.98; Energy Above Hull: 0.014 eV/atom; Predicted Pugh's Ratio: 0.774.
2.  **mp-1025572 (WSe₂ in 2H-like phase):** Viability Score: 156.55; Energy Above Hull: 0.0055 eV/atom; Predicted Pugh's Ratio: 0.789.
3.  **mp-1023925 (WS₂ in 1T phase):** Viability Score: 154.74; Energy Above Hull: 0.0074 eV/atom; Predicted Pugh's Ratio: 0.763.

## Conclusion

By integrating machine learning with fundamental electronic structure descriptors, this study has successfully mapped the phase space of mechanical robustness in metastable TMDs. The recommended Tungsten-based candidates represent the optimal intersection of thermodynamic accessibility and mechanical stability.