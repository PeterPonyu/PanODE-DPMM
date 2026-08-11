# VERIFY-EXTERNAL — DPMM-Base re-aggregation

**Date:** 2026-08-10
**Primary file:** `per_variant_external_wilcoxon.csv`
**SHA256:** `383a3fa8cebacb8964fa52e812246c01b58f9e117921815713bccdb9d1af7a29`

## Procedure

Filter rows with `Internal == DPMM-Base`. Do **not** use `pairwise_wilcoxon_external.csv` (Best-* companion) for these asserts.

## Results

| Check | Expected | Observed |
|---|---|---|
| `len(rows)` | 407 | **407** |
| `Significant_005==Yes` & `Mean_diff>0` | 176 | **176** |
| Core rows: `Metric ∈ {NMI,ARI,ASW,DAV}` & `N_pairs==12` | 44 | **44** |
| Core significant wins (same win rule) | 31 | **31** |
| Core win rate `31/44` | 0.705 | **0.705** |

Cross-check: `tables/external_winrate.tex` DPMM-Base row matches (`407`, `176`, `31`, `44`, `0.705`).

Core significant wins by metric (informational; from primary CSV): NMI 9, ARI 9, ASW 7, DAV 6.

## Companion summary identity

`per_variant_external_summary.csv` SHA256 `fb4fffe83c2fdb0b2ab875872c80a51da48a421470e0675c9d0dbb33f8e6754f` matches archive (byte-identical).

## Companion Best-* file

`pairwise_wilcoxon_external.csv` SHA256 `5c62b106770bd21655f1ceda2f74ce2d5c445b90173d54bd206f0b5d7a059acd` — Best-DPMM / Best-Topic screen only; not used for 31/44.
