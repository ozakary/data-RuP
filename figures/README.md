# Figures, Scripts, and Datasets for Manuscript and Supporting Information "*Fine-Tuned Atomistic Foundation Model Uncovers Short-Range Order and Phase Transition in RuP Superconductor*"
---
📄 Author: **Ouail Zakary**
- 📧 Email: [Ouail.Zakary@oulu.fi](mailto:Ouail.Zakary@oulu.fi)
- 🔗 ORCID: [0000-0002-7793-3306](https://orcid.org/0000-0002-7793-3306)
- 🌐 Website: [Personal Webpage](https://cc.oulu.fi/~nmrwww/members/Ouail_Zakary.html)
- 📁 Portfolio: [Academic Portfolio](https://ozakary.github.io/)
---
This folder contains the figures, Python scripts, datasets, and additional metadata (README files) used to generate the figures in the manuscript and supporting information documents of **"*Fine-Tuned Atomistic Foundation Model Uncovers Short-Range Order and Phase Transition in RuP Superconductor*"**.

## Requirements

To reproduce the figures and analyze the data, you will need the following software:

- **Python 3.x**
- **Matplotlib**
- **NumPy**
- **ASE**
- **Phonopy**
- **MACE**
- **dynasor**
- **SciPy**
- **Seaborn**
- **tqdm**
- **pandas**
- **VESTA** (crystal structure visualization)
- **Inkscape** (figure assembly)

## Table 1: Figures from the Manuscript

| **Figure** | **Script** | **Dataset** | **Readme** |
|------------|------------|-------------|------------|
| [Figure 1](./figure_1.png) | [Script A](./fig-1a_code_plot.py), [Script B](./fig-1b_code_compare_v4.py), [Script C](./fig-1c_code_compare_v4.py), [Script D](./fig-1d_code_compare_v4.py) | [Data (Zenodo)](https://doi.org/TBA) | [README](./figure_1_readme.txt) |
| [Figure 2](./figure_2.png) | `N/A` (VESTA + Inkscape) | `N/A` | [README](./figure_2_readme.txt) |
| [Figure 3](./figure_3.png) | [Script](./fig-3_code_plots_unit_cell_norm_3.py) | [Data (Zenodo)](https://doi.org/TBA) | [README](./figure_3_readme.txt) |
| [Figure 4](./figure_4.png) | [Script A (RDF map)](../rdfs_and_structure-factors/rdf_analysis/code_plot_rdf_ru-ru_opt_with_inset_2d-map.py), [Script B (avg RDF map)](../rdfs_and_structure-factors/avg_rdf_analysis/code_plot_rdf_ru-ru_opt_with_inset_2d-map.py), [Script C (S(q) insta)](../rdfs_and_structure-factors/structure_factor/code_sq_plot.py), [Script D (S(q) avg)](../rdfs_and_structure-factors/avg_structure_factor/code_sq_plot_avg.py) | [Data (Zenodo)](https://doi.org/TBA) | [README](./figure_4_readme.txt) |
| [Figure 5](./figure_5.png) | [Script A (Ru–Ru distances)](../local-order_metrics/ru-ru_along-110-direction_vs_time_and_distr/code_distances_plot_all-temps_kde_2d-map_vf3.py), [Script B (dimer bond order)](../local-order_metrics/dimer_bond_order/code_dimer_bond_order_plot_vf.py) | [Data (Zenodo)](https://doi.org/TBA) | [README](./figure_5_readme.txt) |
| [Figure 6](./figure_6.png) | [Script A (trimer angle)](../local-order_metrics/trimer_angle_order_dashed-curves/code_trimer_angle_order_plot_vf.py), [Script B (trimer bond order)](../local-order_metrics/trimer_bond_order/code_trimer_bond_order_plot_vf.py) | [Data (Zenodo)](https://doi.org/TBA) | [README](./figure_6_readme.txt) |
| [Figure 7](./figure_7.png) | [Script A (time corr.)](../local-order_metrics/correlation_functions/time_correlation/plot_time_correlation.py), [Script B (space corr. parallel)](../local-order_metrics/correlation_functions/space_correlation/plot_space_correlation.py), [Script C (space corr. perpendicular)](../local-order_metrics/correlation_functions/space_correlation/plot_space_correlation.py) | [Data (Zenodo)](https://doi.org/TBA) | [README](./figure_7_readme.txt) |

## Table 2: Figures from the Supporting Information

| **Figure** | **Script** | **Dataset** | **Readme** |
|------------|------------|-------------|------------|
| [Figure S1](./figure_s1.png) | [Script A (energy)](./fig-s1-a_code_plot_energy.py), [Script B (forces)](./fig-s1-b_code_plot_force.py), [Script C (stress)](./fig-s1-c_code_plot_stress.py) | [Data (Zenodo)](https://doi.org/TBA) | [README](./figure_s1_readme.txt) |
| [Figure S2](./figure_s2.png) | [Script](./fig-1b_code_compare_v4.py) | [Data (Zenodo)](https://doi.org/TBA) | [README](./figure_s2_readme.txt) |
| [Figure S3](./figure_s3.png) | [Script](./fig-1b_code_compare_v4.py) | [Data (Zenodo)](https://doi.org/TBA) | [README](./figure_s3_readme.txt) |
| [Figure S4](./figure_s4.png) | [Script](./fig-1b_code_compare_v4.py) | [Data (Zenodo)](https://doi.org/TBA) | [README](./figure_s4_readme.txt) |
| [Figure S5](./figure_s5.png) | [Script](./fig-1c_code_compare_v4.py) | [Data (Zenodo)](https://doi.org/TBA) | [README](./figure_s5_readme.txt) |
| [Figure S6](./figure_s6.png) | [Script](./fig-1c_code_compare_v4.py) | [Data (Zenodo)](https://doi.org/TBA) | [README](./figure_s6_readme.txt) |
| [Figure S7](./figure_s7.png) | [Script](./fig-1c_code_compare_v4.py) | [Data (Zenodo)](https://doi.org/TBA) | [README](./figure_s7_readme.txt) |
| [Figure S8](./figure_s8.png) | [Script A (energy)](./fig-s8-a_code_plot_energy.py), [Script B (forces)](./fig-s8-b_code_plot_force.py), [Script C (stress)](./fig-s8-c_code_plot_stress.py) | [Data (Zenodo)](https://doi.org/TBA) | [README](./figure_s8_readme.txt) |
| [Figure S9](./figure_s9.png) | [Script](./fig-1d_code_compare_v4.py) | [Data (Zenodo)](https://doi.org/TBA) | [README](./figure_s9_readme.txt) |
| [Figure S10](./figure_s10.png) | [Script](./fig-1d_code_compare_v4.py) | [Data (Zenodo)](https://doi.org/TBA) | [README](./figure_s10_readme.txt) |
| [Figure S11](./figure_s11.png) | [Script](./fig-1d_code_compare_v4.py) | [Data (Zenodo)](https://doi.org/TBA) | [README](./figure_s11_readme.txt) |
| [Figure S12](./figure_s12.png) | [Script](../ml-phonons/5-plot_band_structure_improved.py) | [Data (Zenodo)](https://doi.org/TBA) | [README](./figure_s12_readme.txt) |

---

## README Files (Figure Details)

### Figure 1 (`figure_1_readme.txt`)
Figure 1 is composed of four panels (1a–1d). Panel 1a shows the distributions of total energy, atomic forces, and stress tensor components from the training dataset, generated using `fig-1a_code_plot.py`. Panel 1b shows the DFT vs. ML force correlation plot for the Allegro MLIP, generated using `fig-1b_code_compare_v4.py`. Panel 1c shows the DFT vs. ML energy correlation plot for MACE-MP-0b3, generated using `fig-1c_code_compare_v4.py`. Panel 1d shows the DFT vs. ML energy correlation plot for the fine-tuned MACE-MP-0b3, generated using `fig-1d_code_compare_v4.py`. The boxes, arrows, and final assembly were done using **Inkscape**.

### Figure 2 (`figure_2_readme.txt`)
Crystal structure visualizations of the monoclinic and orthorhombic phases of RuP (panels a–e) were produced using **VESTA**. The individual panels were arranged and annotated using **Inkscape** to form the complete figure.

### Figure 3 (`figure_3_readme.txt`)
Panels 3a and 3b (cell parameter differences Δℓ and Δθ vs. temperature) were both generated using `fig-3_code_plots_unit_cell_norm_3.py`. The two panels were arranged in **Inkscape** to form the complete figure.

### Figure 4 (`figure_4_readme.txt`)
Panels 4a and 4b (illustrative snapshots of instantaneous and time-averaged crystal structures) were created by combining **VESTA** (crystal structure visualization) and **Inkscape** (arrows and text). Panels 4c and 4e (2D RDF heatmaps from the MD trajectory and from the time-averaged structures, respectively) were generated using `code_plot_rdf_ru-ru_opt_with_inset_2d-map.py` from `rdfs_and_structure-factors/rdf_analysis/` and `rdfs_and_structure-factors/avg_rdf_analysis/`, respectively. Panels 4d and 4f (structure factors from the MD trajectory and from the time-averaged structures, respectively) were generated using `code_sq_plot.py` and `code_sq_plot_avg.py` from `rdfs_and_structure-factors/structure_factor/` and `rdfs_and_structure-factors/avg_structure_factor/`, respectively. All panels were assembled using **Inkscape**.

### Figure 5 (`figure_5_readme.txt`)
Panel 5a (Ru–Ru distance evolution along [110] vs. temperature) and panel 5b (dimer bond order evolution vs. temperature) were generated using `code_distances_plot_all-temps_kde_2d-map_vf3.py` and `code_dimer_bond_order_plot_vf.py` from `local-order_metrics/`, respectively. The KDE projection panels (50 K and 600 K), the crystal structure visualizations (**VESTA**), and the arrows and text (**Inkscape**) were all assembled in **Inkscape** to form the complete figure.

### Figure 6 (`figure_6_readme.txt`)
Panel 6a (trimer angle evolution vs. temperature) and panel 6b (trimer bond order evolution vs. temperature) were generated using `code_trimer_angle_order_plot_vf.py` and `code_trimer_bond_order_plot_vf.py` from `local-order_metrics/`, respectively. The KDE projection panels (50 K and 600 K), the crystal structure visualizations (**VESTA**), and the arrows and text (**Inkscape**) were all assembled in **Inkscape** to form the complete figure.

### Figure 7 (`figure_7_readme.txt`)
Panel 7a (time auto-correlation function C(t)), panel 7d (relaxation time τ), and panel 7e (oscillation period t_osc) were generated using `plot_time_correlation.py` from `local-order_metrics/correlation_functions/time_correlation/`. Panels 7b and 7c (parallel C∥(r) and perpendicular C⊥(r) space correlation functions) and panels 7f and 7g (parallel ξ∥ and perpendicular ξ⊥ correlation lengths) were generated using `plot_space_correlation.py` from `local-order_metrics/correlation_functions/space_correlation/`. All panels were assembled in **Inkscape** to form the complete figure.

### Figure S1 (`figure_s1_readme.txt`)
Panels S1a, S1b, and S1c (training and validation progress for energy, forces, and stress from the Allegro MLIP) were generated using `fig-s1-a_code_plot_energy.py`, `fig-s1-b_code_plot_force.py`, and `fig-s1-c_code_plot_stress.py`, respectively. The three panels were assembled in **Inkscape** to form the complete figure.

### Figures S2, S3, and S4 (`figure_s2_readme.txt`, `figure_s3_readme.txt`, `figure_s4_readme.txt`)
All correlation panels for Figures S2 (energy: E_monoclinic, E_orthorhombic, stress), S3 (force components: f_x, f_y, f_z), and S4 (stress tensor components: s_xx, s_yy, s_zz, s_xy, s_xz, s_yz) are DFT vs. ML (Allegro) correlation plots, all generated using `fig-1b_code_compare_v4.py`. For each figure, the individual panels were assembled in **Inkscape**.

### Figures S5, S6, and S7 (`figure_s5_readme.txt`, `figure_s6_readme.txt`, `figure_s7_readme.txt`)
All correlation panels for Figures S5 (energy: E_monoclinic, E_orthorhombic, stress), S6 (force components: f_x, f_y, f_z), and S7 (stress tensor components: s_xx, s_yy, s_zz, s_xy, s_xz, s_yz) are DFT vs. ML (MACE-MP-0b3) correlation plots, all generated using `fig-1c_code_compare_v4.py`. For each figure, the individual panels were assembled in **Inkscape**.

### Figure S8 (`figure_s8_readme.txt`)
Panels S8a, S8b, and S8c (training and validation progress for energy, forces, and stress from the fine-tuning of MACE-MP-0b3) were generated using `fig-s8-a_code_plot_energy.py`, `fig-s8-b_code_plot_force.py`, and `fig-s8-c_code_plot_stress.py`, respectively. The three panels were assembled in **Inkscape** to form the complete figure.

### Figures S9, S10, and S11 (`figure_s9_readme.txt`, `figure_s10_readme.txt`, `figure_s11_readme.txt`)
All correlation panels for Figures S9 (energy: E_monoclinic, E_orthorhombic, stress), S10 (force components: f_x, f_y, f_z), and S11 (stress tensor components: s_xx, s_yy, s_zz, s_xy, s_xz, s_yz) are DFT vs. ML (fine-tuned MACE-MP-0b3) correlation plots, all generated using `fig-1d_code_compare_v4.py`. For each figure, the individual panels were assembled in **Inkscape**.

### Figure S12 (`figure_s12_readme.txt`)
Figure S12 consists of 23 panels (S12a–S12w), one per simulation temperature (10 K through 700 K), each showing the temperature-dependent phonon dispersion of the time-averaged monoclinic structure of RuP with an inset phonon DOS. All panels were generated using `5-plot_band_structure_improved.py` from the `ml-phonons/` directory, reading the pre-computed `band_data_{T}K.npz` files. The 23 panels were assembled in **Inkscape** to form the complete figure.

---

### Highlights:
- Links to all figures, scripts, datasets, and README files.
- For each figure, if a script or dataset is not applicable, it is indicated as `N/A`.
- Organized figures into two categories: Manuscript and Supporting Information for clarity.
- Scripts shared across multiple figures are linked individually in each row for clarity.
- Dataset DOIs (Zenodo) are marked as TBA and will be updated upon publication.

Feel free to browse through the figures and check the corresponding scripts and datasets. Ensure all dependencies are installed before attempting to run the scripts for reproducibility.

---
For further details, please refer to the respective folders or contact the author via the provided email.
