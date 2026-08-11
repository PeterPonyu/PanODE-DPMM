# PanODE-DPMM Figure Manifest

This is the submission figure plan. It is intentionally limited to ten figures with four panels each. Existing trained outputs are reused; `No` in the compute column means composition, table rendering, or plotting from saved CSV/NPZ/PT assets only.

| Figure | Panel | Source | Builder | Claim | New compute |
|---|---|---|---|---|---|
| F1 Architecture | A | `models/dpmm_base.py` | `refined_figures/fig01_architecture.py` | Shared Pure-AE backbone | No |
| F1 Architecture | B | `models/dpmm_base.py` | `refined_figures/fig01_architecture.py` | Online DPMM refit after warmup | No |
| F1 Architecture | C | `models/dpmm_flow_matching.py` and DPMM-FM experiment config | `refined_figures/fig01_architecture.py` | Conditional flow transports latents toward mixture centres | No |
| F1 Architecture | D | `refined_figures/output/dpmm/Fig1_architecture_dpmm.pdf` | `refined_figures/fig01_architecture.py` | Three task-dependent operating points | No |
| F2 Geometry-concordance trade-off | A | `article/dpmm/tables/variant_ranking.tex` | `refined_figures/fig02_base_ablation.py` | Pure-AE leads NMI/ARI | No |
| F2 Geometry-concordance trade-off | B | `article/dpmm/tables/variant_ranking.tex` | `refined_figures/fig02_base_ablation.py` | DPMM-Base leads ASW/DAV | No |
| F2 Geometry-concordance trade-off | C | `experiments/results/dpmm_fm_core/summary.csv` | `refined_figures/fig02_base_ablation.py` | DPMM-FM leads projection fidelity | No |
| F2 Geometry-concordance trade-off | D | `article/dpmm/tables/variant_ranking.tex` | `refined_figures/fig02_base_ablation.py` | Variants form a Pareto front, not a dominance chain | No |
| F3 Core statistical evidence | A | `article/dpmm/tables/wilcoxon_core.tex` | table-to-panel composition | ASW improves on 12/12 core datasets | No |
| F3 Core statistical evidence | B | `article/dpmm/tables/wilcoxon_core.tex` | table-to-panel composition | DAV improves on 12/12 core datasets | No |
| F3 Core statistical evidence | C | `article/dpmm/tables/wilcoxon_core.tex` | table-to-panel composition | NMI losses are directional but non-significant | No |
| F3 Core statistical evidence | D | `article/dpmm/tables/wilcoxon_core.tex` | table-to-panel composition | ARI losses are directional but non-significant | No |
| F4 Sensitivity | A | `experiments/results/dpmm_fm_sensitivity_core/summary.csv` | `refined_figures/build_submission_figures.py` | Flow-weight effect on NMI across eight datasets | No |
| F4 Sensitivity | B | same | same | Flow-weight effect on DRE-UMAP | No |
| F4 Sensitivity | C | same | same | Noise-scale effect on ARI | No |
| F4 Sensitivity | D | same | same | Noise-scale effect on LSE | No |
| F5 Training dynamics | A | `benchmarks/training_dynamics_results/DPMM-Base_setty_history.json` | `refined_figures/build_submission_figures.py` | DPMM-Base total-loss trajectory | No |
| F5 Training dynamics | B | same | same | Reconstruction-loss trajectory | No |
| F5 Training dynamics | C | same | same | DPMM loss activates after warmup | No |
| F5 Training dynamics | D | same | same | Snapshot NMI/ARI/ASW trajectories | No |
| F6 Biological exploration | A | `benchmarks/biological_validation/results/*importance*` | `refined_figures/fig04_biological_full.py` | Perturbation-importance workflow | No |
| F6 Biological exploration | B | `benchmarks/biological_validation/results/*correlation*` | same | Latent-gene association structure | No |
| F6 Biological exploration | C | `benchmarks/biological_validation/results/*enrichment*` | same | Components map to coherent GO programs | No |
| F6 Biological exploration | D | `benchmarks/biological_validation/results/*umap*` | same | Three-dataset qualitative grounding; not universal validation | No |
| F7 External benchmark selection | A | `article/dpmm/data/per_variant_external_summary.csv` | `refined_figures/build_std_figures.R` / `build_submission_figures.py` | Full screen: 56 datasets × 18 baselines | No |
| F7 External benchmark selection | B | `article/dpmm/data/per_variant_external_summary.csv` | same | Display subset: 12 core datasets × 11 representative baselines | No |
| F7 External benchmark selection | C | `article/dpmm/data/per_variant_external_summary.csv` | same | Geometry-family comparison | No |
| F7 External benchmark selection | D | `article/dpmm/data/per_variant_external_wilcoxon.csv`; `PROVENANCE-GAP.md` | same | Primary Wilcoxon export in-repo; DPMM-Base 31/44 | No |
| F8 External metric decomposition | A | `article/dpmm/data/per_variant_external_summary.csv` | same | Shared catalogue win rates (Topic-* = sister context only) | No |
| F8 External metric decomposition | B | same | same | All vs core significant wins | No |
| F8 External metric decomposition | C | same | same | Test-count coverage | No |
| F8 External metric decomposition | D | `article/dpmm/data/per_variant_external_wilcoxon.csv`; `PROVENANCE-GAP.md` | same | Provenance closed; core 31/44 regenerable; raw dumps residual | No |
| F9 Downstream utility and cost | A | `article/dpmm/tables/knn.tex` | table-to-panel composition | Pure-AE kNN accuracy 0.784 vs DPMM-Base 0.725 | No |
| F9 Downstream utility and cost | B | same | same | Macro-F1 0.572 vs 0.485 | No |
| F9 Downstream utility and cost | C | `article/dpmm/tables/runtime.tex` | table-to-panel composition | Training time 46.4 s vs 50.5 s | No |
| F9 Downstream utility and cost | D | same | same | Parameter count is unchanged | No |
| F10 Decision map | A | F2, F3, and `variant_ranking.tex` | manuscript composition | Select Pure-AE for label recovery | No |
| F10 Decision map | B | F2, F3, and `variant_ranking.tex` | manuscript composition | Select DPMM-Base for compact geometry | No |
| F10 Decision map | C | `experiments/results/dpmm_fm_core/summary.csv` | manuscript composition | Select DPMM-FM for projection fidelity | No |
| F10 Decision map | D | F6-F9 evidence | manuscript composition | Scope recommendations by downstream task | No |

Provenance boundary: F7–F8 draw on the in-repo primary `per_variant_external_wilcoxon.csv` (SHA256 `383a3fa8…`), companion Best-* `pairwise_wilcoxon_external.csv`, and canonical summary (`fb4fffe8…`). Wilcoxon→summary path is CLOSED per `PROVENANCE-GAP.md` / `data/VERIFY-EXTERNAL.md` (407/176/31/44). Residual: raw per-dataset score dumps not imported. No new training is required.

Build evidence: `article/dpmm/main_mdpi.pdf` is covered by `.gitignore` (repo policy), so the rebuilt PDF is not committed. Rebuild evidence: compiled 2026-08-10 from current sources after provenance import + sister-cite (`./article/build.sh`), 17 pages, 392,401 B, 0 undefined refs; recompile with the same script to reproduce.
