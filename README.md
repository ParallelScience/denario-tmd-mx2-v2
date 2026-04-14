# denario-tmd-mx2-v2

**Scientist:** denario-6
**Date:** 2026-04-14

# Data Description: MX₂ Transition-Metal Dichalcogenides

## Overview

This dataset contains 202 binary MX₂ compounds (one transition metal M, one chalcogen X, in a 1:2 stoichiometric ratio) retrieved from the Materials Project database (https://materialsproject.org) using the `mp-api` Python client (v0.46.0). All values are the result of density functional theory (DFT) calculations performed by the Materials Project using the Vienna Ab initio Simulation Package (VASP) with the PBE generalized-gradient approximation exchange-correlation functional.

The dataset covers 13 transition metals (M = Ti, Zr, Hf, V, Nb, Ta, Cr, Mo, W, Mn, Fe, Co, Ni) and 3 chalcogens (X = S, Se, Te), spanning 35 space groups and 7 crystal systems.

---

## File Location

```
/home/node/work/projects/materials_project_v1/tmd_data_enriched.csv
```

## Loading the Data

```python
import pandas as pd

df = pd.read_csv("/home/node/work/projects/materials_project_v1/tmd_data_enriched.csv")
print(df.shape)   # (202, 53)
```

No index column is stored. All columns are loaded directly.

---

## Column Descriptions

### Identifiers

| Column | Type | Description |
|--------|------|-------------|
| `material_id` | str | Unique Materials Project identifier (e.g. `mp-27793`) |
| `formula` | str | Reduced chemical formula (e.g. `NbSe2`, `MoS2`) |
| `metal` | str | Transition metal element symbol (one of: Ti, Zr, Hf, V, Nb, Ta, Cr, Mo, W, Mn, Fe, Co, Ni) |
| `chalcogen` | str | Chalcogen element symbol (one of: S, Se, Te) |

### Crystal Structure

| Column | Type | Units | Description |
|--------|------|-------|-------------|
| `spacegroup_symbol` | str | — | Hermann-Mauguin space group symbol (e.g. `P6_3/mmc`, `P-3m1`) |
| `spacegroup_number` | int | — | International space group number (1–230) |
| `crystal_system` | str | — | Crystal system: Trigonal, Hexagonal, Orthorhombic, Cubic, Monoclinic, Tetragonal, or Triclinic |
| `phase` | str | — | TMD polymorph classification derived from space group (see Phase Classification below) |
| `volume` | float | Å³ | Total unit cell volume |
| `nsites` | int | — | Number of atomic sites in the unit cell |
| `volume_per_atom` | float | Å³/atom | Unit cell volume divided by number of sites |
| `a`, `b`, `c` | float | Å | Lattice parameters (lengths) |
| `alpha`, `beta`, `gamma` | float | degrees | Lattice parameters (angles) |
| `c_a_ratio` | float | — | Ratio c/a of lattice parameters |

### Thermodynamic Stability

| Column | Type | Units | Description |
|--------|------|-------|-------------|
| `energy_above_hull` | float | eV/atom | Energy above the convex hull. 0 = thermodynamically stable ground state; larger values indicate metastability |
| `log1p_energy_above_hull` | float | — | log(1 + energy_above_hull); log-transformed stability for use in models |
| `formation_energy_per_atom` | float | eV/atom | Formation energy per atom relative to elemental references; negative = exothermic formation |
| `is_stable` | bool | — | True if the material lies on the convex hull (energy_above_hull = 0) |
| `theoretical` | bool | — | True if the material has not been experimentally observed in the ICSD; False if it has been synthesized |

### Electronic Structure

| Column | Type | Units | Description |
|--------|------|-------|-------------|
| `band_gap` | float | eV | DFT-computed electronic band gap. 0.0 = metallic |
| `efermi` | float | eV | Fermi energy in the DFT calculation (absolute, not relative to VBM) |
| `dos_at_fermi` | float | states/eV | Total electronic density of states at the Fermi level, computed from the DFT projected DOS. Missing for 16 materials where no DOS calculation is available in MP. |

### Magnetic Properties

| Column | Type | Description |
|--------|------|-------------|
| `magnetic_ordering` | str | DFT-relaxed magnetic ground state: NM (non-magnetic), FM (ferromagnetic), AFM (antiferromagnetic), FiM (ferrimagnetic) |
| `total_magnetization` | float | Total magnetic moment of the unit cell in μ_B |

### Mechanical / Elastic Properties

Available for 90/202 materials (only computed by MP for stable or near-stable structures).

| Column | Type | Units | Description |
|--------|------|-------|-------------|
| `K_vrh` | float | GPa | Bulk modulus (Voigt-Reuss-Hill average) |
| `G_vrh` | float | GPa | Shear modulus (Voigt-Reuss-Hill average) |
| `poisson_ratio` | float | — | Homogeneous Poisson ratio |
| `elastic_anisotropy` | float | — | Universal elastic anisotropy index |
| `debye_temperature` | float | K | Debye temperature, estimated from elastic constants |

### Atomic / Elemental Descriptors

All values are fixed per element and do not depend on the specific crystal structure.

| Column | Type | Units | Description |
|--------|------|-------|-------------|
| `M_val` | int | — | Valence electron count of the metal (e.g. Mo → 6, Nb → 5) |
| `M_Z` | int | — | Atomic number of the metal |
| `M_en` | float | — | Pauling electronegativity of the metal |
| `M_ie1` | float | eV | First ionization energy of the metal atom |
| `M_atomic_radius` | int | pm | Covalent atomic radius of the metal (Cordero et al. 2008) |
| `M_group` | int | — | Periodic table group of the metal (4–10) |
| `M_period` | int | — | Periodic table period of the metal (4, 5, or 6) |
| `M_soc_proxy` | int | — | Spin-orbit coupling strength proxy: Z² of the metal |
| `X_en` | float | — | Pauling electronegativity of the chalcogen |
| `X_ie1` | float | eV | First ionization energy of the chalcogen atom |
| `X_atomic_radius` | int | pm | Covalent atomic radius of the chalcogen (Cordero et al. 2008) |
| `X_period` | int | — | Periodic table period of the chalcogen (3 = S, 4 = Se, 5 = Te) |
| `en_difference` | float | — | X_en − M_en; proxy for M–X bond ionicity |
| `bond_radius_sum` | int | pm | M_atomic_radius + X_atomic_radius; expected M–X bond length |
| `d_count_m4plus` | int | — | d-electron count of the metal assuming M⁴⁺ oxidation state: M_val − 4 |
| `d_band_filling` | float | — | d_count_m4plus / 10; fractional filling of the 10-electron d-band |

### Known Experimental Superconductivity (Literature)

These columns are populated only for formulas that appear in the experimental literature with a measured superconducting transition temperature. All other rows are NaN. Tc values are phase-specific in principle; the assignment here is formula-level.

| Column | Type | Description |
|--------|------|-------------|
| `Tc_exp_K` | float | Experimental superconducting transition temperature in Kelvin |
| `Tc_phase` | str | Polymorph phase for which Tc was measured (e.g. `2H`, `1T`) |
| `Tc_ref` | str | Short literature reference key |
| `is_known_SC` | bool | True if a positive Tc_exp_K is available for this formula |

---

## Phase Classification

The `phase` column classifies each material into a standard TMD polymorph based on its space group number:

| Phase | Space group(s) | Coordination | Description |
|-------|----------------|--------------|-------------|
| `2H` | 194 (P6₃/mmc) | Trigonal prismatic | Hexagonal, 2-layer repeat. Common semiconducting or metallic phase for Groups V–VI. |
| `2H-like` | 176, 186, 187 | Trigonal prismatic-like | Hexagonal space groups structurally similar to 2H. |
| `1T` | 164 (P-3m1) | Octahedral | Trigonal, 1-layer repeat. Typical for Group IV metals. |
| `3R` | 160, 166 (R3m, R-3m) | Trigonal prismatic | Rhombohedral, 3-layer repeat. |
| `1T'` | 11, 12, 63, 65 (C2/m, P2₁/m, Cmce, Cmmm) | Distorted octahedral | Monoclinic distortion of 1T. Associated with Weyl semimetal phases. |
| `CDW/distorted` | 38, 55, 58, 62 | Distorted | Orthorhombic distortions, often associated with charge-density-wave order. |
| `cubic` | 216, 221, 225, 229 | Mixed | Rocksalt-like or zincblende cubic structures. |
| `other` | all remaining | — | 66 materials with space groups outside the above classifications. Crystal system is available for these. |

---

## Dataset Statistics

| Property | Value |
|----------|-------|
| Total materials | 202 |
| Unique formulas | 37 |
| Metals (M) | Ti, Zr, Hf, V, Nb, Ta, Cr, Mo, W, Mn, Fe, Co, Ni (13 total) |
| Chalcogens (X) | S, Se, Te (3 total) |
| Metallic (band_gap = 0) | 152 |
| Thermodynamically stable (is_stable) | 46 |
| Experimentally observed (theoretical = False) | 77 |
| With DOS at Fermi | 186/202 |
| With elastic constants | 90/202 |
| With known experimental Tc | 80 rows (10 unique formulas) |
| Magnetic ordering — NM / FM / FiM / AFM | 120 / 69 / 9 / 4 |

---

## Missing Values

| Column | Missing count | Reason |
|--------|--------------|--------|
| `dos_at_fermi` | 16 | No DOS calculation available in MP for these material IDs |
| `K_vrh`, `G_vrh`, `poisson_ratio`, `elastic_anisotropy`, `debye_temperature` | 112 | Elasticity calculations not performed by MP for metastable structures |
| `Tc_exp_K`, `Tc_phase`, `Tc_ref` | ~122 | No experimental Tc reported in literature for that formula |

---

## Data Provenance

- *Structural and thermodynamic data*: `MPRester.materials.summary.search` (mp-api v0.46.0), retrieved 2026-04-14
- *Density of states*: `MPRester.get_dos_by_material_id`, retrieved 2026-04-14
- *Magnetic ordering and formation energy*: `MPRester.materials.summary.search`, retrieved 2026-04-14
- *Elastic constants*: `MPRester.elasticity.search`, retrieved 2026-04-14
- *Atomic properties* (electronegativity, ionization energy, atomic radius): standard literature values (Pauling scale; Cordero et al. 2008 for radii; NIST Atomic Spectra Database for IEs)
- *Experimental Tc values*: compiled from primary literature (Morris 1972, Revolinsky 1965, Wilson 1975, DiSalvo 1973, Morosan 2006, Costanzo 2016, Shi 2015, Lu 2015, Ye 2012, Qi 2016, Kang 2015)