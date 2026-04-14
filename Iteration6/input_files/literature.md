The proposed idea, "Electronic Phase-Space Mapping of Mechanical Robustness in Metastable TMDs," is novel in its specific application of machine learning to bridge the gap between electronic structure descriptors and mechanical viability in metastable Transition-Metal Dichalcogenides (TMDs). While high-throughput screening of TMDs is a well-established field, existing literature primarily focuses on predicting thermodynamic stability (energy above the hull) or superconducting transition temperatures ($T_c$).

The most relevant papers that share similar methodologies but differ in objective include:

1. **"High-throughput computational screening of transition metal dichalcogenides for electronic and optoelectronic applications"** (https://doi.org/10.1038/s41524-017-0035-x): This study performs high-throughput screening of TMDs but focuses on band gaps and thermodynamic stability rather than mechanical robustness or the classification of metastable phases.
2. **"Machine learning of the elastic properties of materials"** (https://doi.org/10.1103/PhysRevMaterials.2.083801): This work uses machine learning to predict elastic constants (bulk and shear moduli) across broad chemical spaces. 

**Novelty Assessment:**
The novelty of your approach lies in three areas:
* **Classification vs. Regression:** Unlike the cited literature that attempts to regress continuous elastic constants—which is often limited by the scarcity of high-quality elastic data for metastable phases—your approach uses a binary classification framework to identify "mechanically robust" candidates.
* **Feature Engineering for Metastability:** By explicitly incorporating `d_band_filling`, `en_difference`, and `dos_at_fermi` as proxies for electronic-driven structural distortions (like Jahn-Teller effects), you are targeting the specific physical mechanisms that typically destabilize metastable TMDs.
* **Focus on the "Elastic Gap":** Your focus on the 112 metastable candidates that lack computed elastic constants in the Materials Project database addresses a specific data-gap problem, effectively using the 90-sample stable population as a training set to infer mechanical viability in the metastable regime.

While the individual components (TMD datasets, random forest classifiers, and elastic property prediction) are known, the synthesis of these into a "mechanical viability" filter for metastable materials is a distinct and valuable contribution to the field of computational materials discovery.