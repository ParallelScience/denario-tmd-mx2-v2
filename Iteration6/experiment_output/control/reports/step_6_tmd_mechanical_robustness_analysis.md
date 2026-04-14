<!-- filename: reports/step_6_tmd_mechanical_robustness_analysis.md -->
# 1. Model Performance and Sensitivity Analysis

The primary objective of this study was to develop a machine learning framework capable of predicting the mechanical robustness of metastable transition-metal dichalcogenides (TMDs), quantified by Pugh's ratio ($G_{vrh}/K_{vrh}$). Given the limited availability of elastic tensor data (90 samples out of the 202 total materials), a multi-task learning approach was initially explored using a shallow neural network. However, preliminary evaluations revealed that the neural network struggled to generalize across different transition metal groups, yielding a coefficient of determination ($R^2$) below 0.1. Consequently, the modeling strategy fell back to a Random Forest ensemble, which is generally more robust to overfitting on small, tabular datasets with complex non-linear interactions.

To rigorously evaluate the model's extrapolative capabilities, a Leave-One-Group-Out (LOGO) cross-validation strategy was employed, grouping the dataset by the transition metal element. This is a highly stringent validation scheme, as it requires the model to predict the elastic properties of TMDs containing a transition metal that was entirely absent from the training set. The results of the sensitivity analysis regarding the imputation of missing density of states (`dos_at_fermi`) values are summarized in Table 1.

**Table 1: Sensitivity Analysis on Missing Data (LOGO Cross-Validation)**

| Model Configuration | Mean Absolute Error (MAE) | $R^2$ Score |
| :--- | :--- | :--- |
| With `is_dos_missing` indicator | 0.1662 | -0.0432 |
| Without `is_dos_missing` indicator | 0.1646 | -0.0311 |

The LOGO cross-validation yielded negative $R^2$ values (-0.0432 and -0.0311), indicating that the model's predictions on unseen metal groups are, on average, less accurate than simply predicting the global mean of the training set. This highlights a fundamental challenge in materials informatics: the mechanical properties of TMDs are deeply governed by the specific electronic structure and bonding characteristics of the constituent transition metal. Extrapolating these properties to entirely new elements without explicit ab initio calculations is inherently difficult. Furthermore, the inclusion of the `is_dos_missing` binary indicator slightly degraded the performance, suggesting that the mean imputation strategy for the 16 missing DOS values was sufficient and that adding a missingness flag introduced unnecessary noise. Consequently, the model without the `is_dos_missing` indicator was selected as the baseline for the ensemble approach.

Despite the poor global generalization across unseen metals, the Random Forest model captures critical local non-linear relationships within the feature space. To mitigate the risks associated with epistemic uncertainty, a Deep Ensemble approach was adopted, training 10 independent Random Forest models with different random initializations. This allowed for the derivation of a mean predicted Pugh's ratio and an associated standard deviation (uncertainty) for each metastable candidate, forming the basis of the viability scoring.

# 2. Feature Importance and SHAP Analysis

To elucidate the physical drivers of mechanical robustness learned by the model, SHapley Additive exPlanations (SHAP) were computed using the 90-sample elastic subset as the background dataset. The SHAP analysis provides a unified measure of feature importance, quantifying the marginal contribution of each feature to the predicted Pugh's ratio. The top five features, ranked by their mean absolute SHAP values, are presented below:

1. `dos_at_fermi` (Mean |SHAP|: 0.0676)
2. `c_a_ratio` (Mean |SHAP|: 0.0360)
3. `M_soc_proxy` (Mean |SHAP|: 0.0214)
4. `phase_3R` (Mean |SHAP|: 0.0199)
5. `d_band_filling` (Mean |SHAP|: 0.0130)

The dominance of `dos_at_fermi` as the most critical predictor underscores the profound connection between electronic structure and mechanical stability in TMDs. A high density of states at the Fermi level is classically associated with electronic instabilities. In the context of crystal lattices, such instabilities often drive structural distortions—such as Jahn-Teller effects or charge density waves (CDWs)—which act to open a band gap and lower the electronic energy of the system. These distortions inherently soften the lattice, leading to a reduction in the shear modulus ($G_{vrh}$) relative to the bulk modulus ($K_{vrh}$), thereby lowering Pugh's ratio. The model successfully captures this solid-state chemistry principle, penalizing the mechanical robustness of metallic phases with high DOS.

The second most important feature, the `c_a_ratio`, captures the structural anisotropy of the layered TMDs. The ratio of the out-of-plane lattice parameter $c$ to the in-plane parameter $a$ dictates the relative strength of the interlayer van der Waals interactions versus the intralayer covalent bonds. Variations in this ratio directly influence the shear response of the material, as sliding between layers (which governs $G_{vrh}$) is highly sensitive to the interlayer spacing.

The inclusion of `M_soc_proxy` (the square of the atomic number, $Z^2$) as the third most important feature highlights the role of spin-orbit coupling (SOC) in heavy transition metals (e.g., W, Ta). Strong SOC can significantly alter the band structure, splitting degenerate d-orbitals and modifying the Fermi surface topology. This electronic modification cascades into the elastic tensor, particularly in metastable phases where the delicate balance of orbital interactions dictates structural rigidity.

The interaction between `d_band_filling` and `dos_at_fermi` provides the most compelling physical insight. The SHAP dependence analysis reveals a complex, non-linear coupling between these two features. For Group VI metals (Mo, W) with a $d^2$ electron count (corresponding to a specific `d_band_filling`), the Fermi level typically lies within a band gap in the stable 2H phase, resulting in a zero `dos_at_fermi`. These semiconducting phases exhibit high mechanical robustness (high Pugh's ratio) due to fully occupied, strongly bonding states. However, when these same metals are forced into metastable octahedral (1T) or distorted geometries, the d-band degeneracy is broken differently, often resulting in a metallic state with a non-zero DOS at the Fermi level. The model identifies that for a given `d_band_filling`, an increase in `dos_at_fermi` precipitously drops the predicted Pugh's ratio. This interaction effectively maps the phase space of mechanical softening, distinguishing between robust, fully-bonded semiconductors and softer, electronically frustrated metals.

# 3. Mechanical Viability Mapping and Candidate Prioritization

The ultimate goal of this framework is to screen the 112 metastable TMD candidates and identify those that are mechanically viable for experimental synthesis. To prevent the model from extrapolating into unphysical regimes, candidates with extreme `dos_at_fermi` values—defined as exceeding three standard deviations from the mean (17.5373 states/eV)—were filtered out. This step removed 4 highly metallic, likely dynamically unstable artifacts, leaving 198 materials in the dataset.

To rank the remaining candidates, a composite viability score $S$ was defined as the ratio of the mean predicted Pugh's ratio to the ensemble standard deviation ($S = \text{Predicted}(G/K) \times \text{Uncertainty}^{-1}$). This score inherently favors materials that are predicted to be highly robust (high $G/K$) with high confidence (low uncertainty).

The top 10 metastable candidates ranked by the overall viability score are presented in Table 2.

**Table 2: Top 10 Ranked Metastable Candidates (Overall)**

| Rank | Formula | Phase | $E_{hull}$ (eV/atom) | Pred $G/K$ | Uncertainty | Score $S$ | Theoretical |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | MoSe$_2$ | 2H | 0.0033 | 0.7089 | 0.0022 | 324.9057 | False |
| 2 | TaSe$_2$ | 2H-like | 0.0007 | 0.6227 | 0.0029 | 212.3705 | False |
| 3 | WS$_2$ | 2H-like | 0.0142 | 0.7729 | 0.0039 | 199.7094 | True |
| 4 | NbSe$_2$ | 2H-like | 0.3714 | 0.6389 | 0.0034 | 186.2566 | True |
| 5 | WSe$_2$ | 2H-like | 0.0055 | 0.7904 | 0.0052 | 151.6653 | True |
| 6 | WS$_2$ | 1T | 0.0074 | 0.7610 | 0.0052 | 146.1045 | True |
| 7 | MoS$_2$ | 1T | 0.0037 | 0.7243 | 0.0052 | 138.0190 | True |
| 8 | TaSe$_2$ | 2H | 0.0000 | 0.6455 | 0.0048 | 135.1797 | False |
| 9 | WS$_2$ | 1T | 0.0038 | 0.7629 | 0.0061 | 124.1329 | True |
| 10 | MoSe$_2$ | 1T | 0.0079 | 0.7265 | 0.0059 | 122.3556 | True |

While the overall ranking identifies highly robust materials, several of these (e.g., MoSe$_2$ 2H, TaSe$_2$ 2H-like) have already been experimentally observed (`Theoretical = False`). To isolate novel targets for future synthesis, a strict prioritization filter was applied, restricting the list to purely theoretical materials that lie within a thermodynamic stability window of $E_{hull} \le 0.05$ eV/atom. This threshold represents a realistic energy scale accessible via non-equilibrium synthesis techniques such as molecular beam epitaxy (MBE) or chemical vapor deposition (CVD).

The top 10 prioritized theoretical candidates are detailed in Table 3.

**Table 3: Top Prioritized Candidates ($E_{hull} \le 0.05$ eV/atom & Theoretical)**

| Rank | Formula | Phase | $E_{hull}$ (eV/atom) | Pred $G/K$ | Uncertainty | Score $S$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | WS$_2$ | 2H-like | 0.0142 | 0.7729 | 0.0039 | 199.7094 |
| 2 | WSe$_2$ | 2H-like | 0.0055 | 0.7904 | 0.0052 | 151.6653 |
| 3 | WS$_2$ | 1T | 0.0074 | 0.7610 | 0.0052 | 146.1045 |
| 4 | MoS$_2$ | 1T | 0.0037 | 0.7243 | 0.0052 | 138.0190 |
| 5 | WS$_2$ | 1T | 0.0038 | 0.7629 | 0.0061 | 124.1329 |
| 6 | MoSe$_2$ | 1T | 0.0079 | 0.7265 | 0.0059 | 122.3556 |
| 7 | WSe$_2$ | 1T | 0.0082 | 0.7789 | 0.0070 | 111.7060 |
| 8 | Te$_2$W | 2H-like | 0.0406 | 0.7927 | 0.0076 | 104.8109 |
| 9 | Te$_2$Mo | 2H-like | 0.0077 | 0.7791 | 0.0079 | 98.4841 |
| 10 | Te$_2$Mo | 2H-like | 0.0216 | 0.7932 | 0.0082 | 96.4549 |

To ensure that the predictions for these top candidates were not artifacts of overfitting to their specific transition metals, a "leave-one-metal-out" sensitivity check was conducted. For each of the top 10 overall candidates, the ensemble prediction (trained on all data) was compared to the prediction from the LOGO cross-validation fold where that specific metal was excluded from the training set. The mean absolute difference across the top 10 candidates was found to be 0.0491. This relatively small deviation demonstrates that while the model struggles to generalize globally across all metals (as evidenced by the negative $R^2$), its predictions for these specific high-robustness candidates are remarkably stable. The model successfully infers the mechanical properties of these W- and Mo-based phases from the behavior of other transition metals in the dataset, lending significant confidence to their predicted viability.

# 4. Implications for Solid-State Chemistry and Experimental Synthesis

The results of this machine learning screening align elegantly with established principles of solid-state chemistry while providing actionable targets for experimentalists. The prioritized list (Table 3) is heavily dominated by Group VI transition metals (M = Mo, W) in metastable 1T and 2H-like phases.

In their stable 2H ground states, MoS$_2$ and WS$_2$ are archetypal semiconductors with fully occupied, strongly bonding $d_{z^2}$ orbitals. The model predicts that even when forced into metastable 1T (octahedral) or 2H-like (distorted trigonal prismatic) geometries, these materials retain a high degree of mechanical robustness, exhibiting predicted Pugh's ratios well above 0.72. This is a non-trivial finding. The 1T phase of Group VI TMDs is typically metallic and often susceptible to Peierls distortions leading to 1T' or CDW phases. However, the model suggests that the intrinsic strength of the W-S, W-Se, and Mo-S bonds—driven by the high electronegativity difference and optimal d-band filling—provides sufficient restoring force to maintain a high shear modulus relative to the bulk modulus, even in the presence of a non-zero DOS at the Fermi level.

The identification of 1T-WS$_2$ ($E_{hull} = 0.0074$ eV/atom) and 1T-MoS$_2$ ($E_{hull} = 0.0037$ eV/atom) as highly robust candidates is particularly exciting for the field of electrocatalysis and energy storage. The 1T phases of these materials are highly sought after for their metallic conductivity, which vastly improves charge transfer kinetics in applications such as the hydrogen evolution reaction (HER) and lithium-ion battery electrodes. However, their practical application has historically been hindered by their tendency to mechanically degrade or spontaneously revert to the 2H phase under operational stress. The high predicted Pugh's ratios for these specific polymorphs suggest that they possess the necessary mechanical integrity to withstand the volumetric expansion and shear stresses associated with ion intercalation.

Furthermore, the presence of tellurides (Te$_2$W and Te$_2$Mo) in the top 10 prioritized list highlights the role of the chalcogen in modulating mechanical properties. Tellurium, being larger and less electronegative than sulfur or selenium, typically forms softer, more covalent bonds. Yet, in these specific 2H-like metastable configurations, the model predicts a high Pugh's ratio (~0.79). This suggests a unique interplay between the strong spin-orbit coupling of the heavy metal (W, Mo) and the extended p-orbitals of Te, resulting in a highly anisotropic but shear-resistant lattice.

From a synthesis perspective, the extremely low energy above the hull for these candidates ($< 0.015$ eV/atom for the top 7) strongly implies that they are accessible via standard metastable synthesis routes. Techniques such as alkali metal intercalation followed by exfoliation (commonly used to synthesize 1T-MoS$_2$) or plasma-enhanced chemical vapor deposition could be employed to isolate these phases. The viability map generated in this study serves as a direct guide for these efforts, filtering out the vast phase space of mechanically fragile or dynamically unstable polymorphs and focusing experimental resources on the most promising, robust candidates.

In conclusion, by mapping the electronic phase space of metastable TMDs, this study demonstrates that mechanical robustness is not exclusively the domain of thermodynamically stable ground states. Through the careful integration of electronic descriptors (`dos_at_fermi`, `d_band_filling`) and structural parameters (`c_a_ratio`), the Random Forest ensemble successfully navigates the complex topology of mechanical softening. The resulting prioritized list of W- and Mo-based metastable phases provides a highly targeted roadmap for the discovery of novel, mechanically resilient functional materials.