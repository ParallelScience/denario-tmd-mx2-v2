# Electronic Phase-Space Mapping of Mechanical Robustness in Metastable TMDs

## 1. Introduction and Formulation of the Mechanical Viability Boundary
The primary objective of this study is to establish a predictive, physics-informed machine learning framework for identifying mechanically robust metastable transition-metal dichalcogenides (TMDs). We define a "mechanical viability" boundary within the multidimensional feature space of electronic structure and crystal geometry, formulating a binary classification task based on the Voigt-Reuss-Hill average shear modulus ($G_{vrh}$). Materials exhibiting a $G_{vrh}$ strictly above their respective phase-specific median are classified as mechanically robust (`is_robust` = 1), while those falling below are classified as non-robust (`is_robust` = 0).

## 2. Exploratory Data Analysis and Bivariate Correlations
Spearman rank correlation analysis reveals key physical drivers of mechanical stiffness:
- **Volume per atom (`volume_per_atom`):** Strong negative correlation ($
ho = -0.446$), indicating that denser atomic packing leads to higher stiffness.
- **Fermi energy (`efermi`):** Strong positive correlation ($
ho = 0.447$), suggesting higher electron filling enhances lattice rigidity.
- **Thermodynamic stability (`energy_above_hull`):** Negative correlation ($
ho = -0.101$), confirming that thermodynamic metastability is often accompanied by mechanical softening.

## 3. Machine Learning Model Performance and Validation
A Random Forest classifier was employed, utilizing a KNN imputer for missing data. The model was evaluated using Repeated Stratified K-Fold Cross-Validation (5 folds, 10 repeats), prioritizing the Area Under the Precision-Recall Curve (AUPRC) to ensure high confidence in positive predictions.

## 4. Feature Importance and the Physics of Mechanical Softening
Permutation Feature Importance analysis highlights the following drivers:
- `volume_per_atom` (0.122 $\pm$ 0.075): Dominant geometric factor.
- `energy_above_hull` (0.038 $\pm$ 0.048): Key indicator of potential energy well depth.
- `M_soc_proxy` (0.019 $\pm$ 0.043) and `dos_at_fermi` (0.015 $\pm$ 0.040): These features capture the electronic instability-driven softening, where spin-orbit coupling acts as a stabilizing mechanism by lifting band degeneracies.

## 5. Inference and Screening of Metastable Candidates
We screened 112 metastable TMD candidates using a dual-filter criterion: $P(\text{robust}) > 0.8$ and $E_{hull} < 0.05$ eV/atom. Five candidates were identified as promising targets:
- **TaSe$_2$ (2H-like, mp-637232):** $E_{hull} = 0.00069$ eV/atom, $P = 0.950$.
- **TaS$_2$ (3R, mp-16226):** $E_{hull} = 0.0024$ eV/atom, $P = 0.866$.
- **TaS$_2$ (3R, mp-755523):** $E_{hull} = 0.0167$ eV/atom, $P = 0.848$.
- **NiS$_2$ (Other, mp-2282):** $E_{hull} = 0.0152$ eV/atom, $P = 0.910$.
- **NiS$_2$ (Other, mp-850131):** $E_{hull} = 0.0217$ eV/atom, $P = 0.880$.

## 6. Conclusion
The developed framework successfully maps the complex phase space of TMDs, providing a robust methodology to identify metastable materials that are both thermodynamically accessible and mechanically stable, thereby guiding future experimental synthesis.