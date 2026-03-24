# Figures, Scripts, and Datasets for Manuscript and Supporting Information "*Local Symmetry Breaking and Two-Stage Phase Transition in RuP Uncovered by a Fine-Tuned Atomistic Foundation Model*"
---
📄 Author: **Ouail Zakary**
- 📧 Email: [Ouail.Zakary@oulu.fi](mailto:Ouail.Zakary@oulu.fi)
- 🔗 ORCID: [0000-0002-7793-3306](https://orcid.org/0000-0002-7793-3306)
- 🌐 Website: [Personal Webpage](https://cc.oulu.fi/~nmrwww/members/Ouail_Zakary.html)
- 📁 Portfolio: [Academic Portfolio](https://ozakary.github.io/)
---
This folder contains the figures, Python scripts, datasets, and additional metadata (README files) used to generate the figures in the manuscript and supporting information documents of **"*Local Symmetry Breaking and Two-Stage Phase Transitions in RuP Uncovered by a Fine-Tuned Atomistic Foundation Model*"**.

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
| Figure 1 | `N/A` (VESTA + Inkscape) | `N/A` | [README](./figure_1_readme.txt) |
| Figure 2 | [Script](./fig-2_code_plot.py) | [Data (Zenodo)](https://doi.org/10.5281/zenodo.18709769) | [README](./figure_2_readme.txt) |
| Figure 3 | [Script A](./fig-3a_code_compare_v4.py), [Script B](./fig-3b_code_compare_v4.py) | [Data (Zenodo)](https://doi.org/10.5281/zenodo.18709769) | [README](./figure_3_readme.txt) |
| Figure 4 | [Script A](./fig-4ab_code_plots_unit_cell_norm_3.py), [Script B](./fig-4ab_code_plots_unit_cell_norm_3.py), [Script C](./fig-4c_code_compute_symm.py) | [Data (Zenodo)](https://doi.org/10.5281/zenodo.18709769) | [README](./figure_4_readme.txt) |
| Figure 5 | [Script A (RDF map)](../rdfs_and_structure-factors/code_plot_rdf_ru-ru_opt_with_inset_2d-map.py), [Script B (avg RDF map)](../rdfs_and_structure-factors/code_plot_avg_rdf_ru-ru_opt_with_inset_2d-map.py), [Script C (S(q) insta)](../rdfs_and_structure-factors/code_sq_plot.py), [Script D (S(q) avg)](../rdfs_and_structure-factors/code_sq_plot_avg.py) | [Data (Zenodo)](https://doi.org/10.5281/zenodo.18709769) | [README](./figure_5_readme.txt) |
| Figure 6 | [Script A (Ru–Ru distances)](../local-order_metrics/code_distances_plot_all-temps_kde_2d-map_vf3.py), [Script B (dimer bond order)](../local-order_metrics/code_dimer_bond_order_plot_vf.py) | [Data (Zenodo)](https://doi.org/10.5281/zenodo.18709769) | [README](./figure_6_readme.txt) |
| Figure 7 | [Script A (time corr.)](../local-order_metrics/plot_time_correlation.py), [Script B (space corr.)](../local-order_metrics/plot_space_correlation.py) | [Data (Zenodo)](https://doi.org/10.5281/zenodo.18709769) | [README](./figure_7_readme.txt) |

## Table 2: Figures from the Supporting Information

| **Figure** | **Script** | **Dataset** | **Readme** |
|------------|------------|-------------|------------|
| Figure S1 | [Script](./fig-s1_code_plot_all_metrics.py) | [Data (Zenodo)](https://doi.org/10.5281/zenodo.18709769) | [README](./figure_s1_readme.txt) |
| Figure S2 | [Script](./fig-3a_code_compare_v4.py) | [Data (Zenodo)](https://doi.org/10.5281/zenodo.18709769) | [README](./figure_s2_readme.txt) |
| Figure S3 | [Script](./fig-3b_code_compare_v4.py) | [Data (Zenodo)](https://doi.org/10.5281/zenodo.18709769) | [README](./figure_s3_readme.txt) |
| Figure S4 | [Script](./fig-3a_code_compare_v4.py) | [Data (Zenodo)](https://doi.org/10.5281/zenodo.18709769) | [README](./figure_s4_readme.txt) |
| Figure S5 | [Script](./fig-3b_code_compare_v4.py) | [Data (Zenodo)](https://doi.org/10.5281/zenodo.18709769) | [README](./figure_s5_readme.txt) |
| Figure S6 | [Script A (trimer angle)](../local-order_metrics/code_trimer_angle_order_plot_vf.py), [Script B (trimer bond order)](../local-order_metrics/code_trimer_bond_order_plot_vf.py) | [Data (Zenodo)](https://doi.org/10.5281/zenodo.18709769) | [README](./figure_s6_readme.txt) |
| Figure S7 | [Script](../local-order_metrics/plot_time_correlation.py) | [Data (Zenodo)](https://doi.org/10.5281/zenodo.18709769) | [README](./figure_s7_readme.txt) |
| Figure S8 | [Script](../local-order_metrics/plot_time_correlation.py) | [Data (Zenodo)](https://doi.org/10.5281/zenodo.18709769) | [README](./figure_s8_readme.txt) |
| Figure S9 | [Script](../local-order_metrics/plot_time_correlation.py) | [Data (Zenodo)](https://doi.org/10.5281/zenodo.18709769) | [README](./figure_s9_readme.txt) |
| Figure S10 | [Script](../local-order_metrics/plot_time_correlation.py) | [Data (Zenodo)](https://doi.org/10.5281/zenodo.18709769) | [README](./figure_s10_readme.txt) |
| Figure S11 | [Script](../local-order_metrics/plot_time_correlation.py) | [Data (Zenodo)](https://doi.org/10.5281/zenodo.18709769) | [README](./figure_s11_readme.txt) |
| Figure S12 | [Script](../local-order_metrics/plot_space_correlation.py) | [Data (Zenodo)](https://doi.org/10.5281/zenodo.18709769) | [README](./figure_s12_readme.txt) |
| Figure S13 | [Script](../ml-phonons/5-plot_band_structure_improved.py) | [Data (Zenodo)](https://doi.org/10.5281/zenodo.18709769) | [README](./figure_s13_readme.txt) |


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
