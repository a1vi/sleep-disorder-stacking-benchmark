# Final Elsevier Manuscript Package (NEURI-D-26-00299)

This directory contains the self-contained LaTeX source, number macros, table definitions, figures, reviewer response letter, and compiled preview for the revised manuscript:

> **"An Explainable Stacking Framework for Multiclass Sleep-Disorder Risk Stratification"**  
> **Authors:** Rifat Bin Reza, Md Ahadul Haque Suvo, Atair Rahman Alvi  
> **Elsevier Reference:** `NEURI-D-26-00299`

---

## Package Directory Contents

### 1. Main Manuscript Files
- **`Final_elsevier_revised.tex`**: Master LaTeX source file using Elsevier's `cas-dc` (two-column) class with embedded `thebibliography`.
- **`Final_elsevier_revised_preview.pdf`**: Complete compiled PDF of the revised manuscript.

### 2. Macros and Table Inputs
- **`numbers.tex`**: Macro definitions for all empirical results, sample counts, and metrics reported in the paper.
- **`extra_numbers.tex`**: Additional macros for ablation experiments.
- **`tab_orig.tex`**: Table 2 — Single hold-out partition performance (75 records).
- **`tab_main.tex`**: Table 3 — Main benchmark: 10×5 repeated grouped cross-validation across 13 models.
- **`tab_nested.tex`**: Table 4 — Nested cross-validation results.
- **`tab_paired.tex`**: Table 5 — Paired difference statistics and McNemar test results.
- **`tab_shap.tex`**: Table 6 — SHAP explanation stability and rank correlations.
- **`tab_lime.tex`**: Table 7 — LIME explanation audit and sample fidelity.
- **`tab_stackabl.tex`**: Table 8 — Meta-learner ablation configurations.
- **`tab_ablation.tex`**: Table 9 — Feature group ablation results.
- **`tab_dedup.tex`**: Deduplicated dataset (132 unique records) benchmark table.

### 3. Reviewer Response Files
- **`Response_to_Reviewers.tex`**: Point-by-point response to reviewers source.
- **`Response_to_Reviewers.pdf`**: Compiled point-by-point response document.

### 4. Figures (`figs/`)
All figures referenced by `Final_elsevier_revised.tex` in vector PDF and raster PNG formats:
- `archi_fig.jpg`: Workflow overview (6 stages).
- `fig1_protocol.png` / `fig1_protocol.pdf`: Leakage isolation protocol.
- `fig2_audit.png` / `fig2_audit.pdf`: Dataset audit & 93.3% Bayes ceiling.
- `fig3_models.png` / `fig3_models.pdf`: 13-model benchmark forest plots.
- `fig4_paired.png` / `fig4_paired.pdf`: Paired difference comparisons vs Stacking.
- `fig5_calibration.png` / `fig5_calibration.pdf`: Reliability diagrams & Brier score.
- `fig6_shap.png` / `fig6_shap.pdf`: SHAP stability across 10 runs.
- `pca_class_separability.png`: 2D PCA projection with 1-sigma class ellipses.
- `randomforest.png`: Heterogeneous stacking ensemble architecture.
- `feature_relationships_ale.png`: Spearman correlation & 2D ALE surface.
- `stacking_roc_auc.png`: Multiclass ROC / AUC curves.
- `confusion_matrices.png`: Hold-out confusion matrices.
- `voting_shap_importance.png`: Global SHAP feature importance.
- `voting_lime_sample0.png`: Local LIME explanation for sample #0.
- `567.png`: Local LIME explanation for Patient #244.
- `shap_lime_radar.png`: SHAP vs LIME radar attribution audit.
- `stacking_shap_beeswarm_insomnia.png`: Insomnia SHAP beeswarm plot.
- `output.png`: Additional attribution diagnostics.
- `12.png`: Random-split sensitivity analysis across seeds.

---

## Compilation Instructions

To compile the manuscript locally using `pdflatex`:

```bash
pdflatex Final_elsevier_revised.tex
pdflatex Final_elsevier_revised.tex
```

*Note: The document class is Elsevier's standard `cas-dc` (`\documentclass[a4paper,fleqn]{cas-dc}`). When compiling on Elsevier Editorial Manager or on a system with TeX Live / MiKTeX, the `els-cas-templates` package is utilized.*
