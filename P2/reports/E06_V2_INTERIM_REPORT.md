# E06 — cands_BCD_v2 Modeling Interim Report

**Experiment**: E06 — LightGBM Modeling on V2 Candidate Universe  
**Date**: 2026-09-26  
**Status**: COMPLETE (Candidate generation, 5-fold training, Full 67.3M OOF generation, Primary T=0.50 scoring complete; Threshold sweep intentionally stopped at partial diagnostic stage)

---

## 1. Executive Summary

E06 successfully converted the `cands_BCD_v2` candidate universe expansion into actual model score gains under the frozen E02 modeling recipe:

- **V2 Candidate Oracle**: **`0.827200935`** (Ceiling gain vs V1 Oracle `0.784522591`: **`+0.042678344`**)
- **V2 Actual Score @ $T=0.50$ (Primary Controlled Baseline)**: **`0.747084844`**
- **E02 Baseline Score @ $T=0.50$ (V1)**: `0.718765173`
- **Primary Macro F0.5 Gain over E02**: **`+0.028319671` (+2.83 Macro F0.5 points)**

> **Important Note on Threshold Sweep**:  
> *Full 0.500–0.995 sweep was intentionally stopped; reported threshold results are the completed evaluations available before stop. Best global threshold is not selected from this partial run; nested fold calibration (E07) is recommended for unbiased decision rules.*

---

## 2. Completed Artifacts Inventory

All generated model binaries and the full canonical OOF artifact are intact on disk:

| Artifact | Path | Size | Rows / Details |
|---|---|---|---|
| **Full Canonical V2 OOF** | `P2/reports/E06_V2_OOF_FULL_CANONICAL.tsv` | 2,928,257,277 bytes (~2.73 GB) | **67,332,524 candidate pairs** |
| **Fold 0 LightGBM Model** | `P2/models/e06_v2_lgb_fold0.txt` | 1,797,447 bytes | 500 trees, 31 leaves |
| **Fold 1 LightGBM Model** | `P2/models/e06_v2_lgb_fold1.txt` | 1,797,233 bytes | 500 trees, 31 leaves |
| **Fold 2 LightGBM Model** | `P2/models/e06_v2_lgb_fold2.txt` | 1,797,082 bytes | 500 trees, 31 leaves |
| **Fold 3 LightGBM Model** | `P2/models/e06_v2_lgb_fold3.txt` | 1,796,394 bytes | 500 trees, 31 leaves |
| **Fold 4 LightGBM Model** | `P2/models/e06_v2_lgb_fold4.txt` | 1,796,997 bytes | 500 trees, 31 leaves |
| **Interim Results TSV** | `P2/reports/E06_V2_INTERIM_RESULTS.tsv` | ~850 bytes | Scored oracle & threshold metrics |

---

## 3. V2 Candidate Volume & Verification (Task 1)

All generated candidate pairs match the exact universe specification with zero duplicates:

| Split | File | Expected Rows | Verified Rows | Duplicates | Status |
|---|---|---|---|---|---|
| **Train S2** | `P2/data/candidates/train_candidate_pairs_s2_v2.tsv` | 30,359,040 | 30,359,040 | 0 | **OK** |
| **Train S3** | `P2/data/candidates/train_candidate_pairs_s3_v2.tsv` | 36,973,484 | 36,973,484 | 0 | **OK** |
| **Train Total** | — | **67,332,524** | **67,332,524** | **0** | **OK** |
| **Test S2** | `P2/data/candidates/test_candidate_pairs_s2_v2.tsv` | 34,919,169 | 34,919,169 | 0 | **OK** |
| **Test S3** | `P2/data/candidates/test_candidate_pairs_s3_v2.tsv` | 41,714,627 | 41,714,627 | 0 | **OK** |
| **Test Total** | — | **76,633,796** | **76,633,796** | **0** | **OK** |

---

## 4. Model Training & Runtimes (Tasks 2 & 3)

- **Training Sample**: 11,257,634 rows (5,023,168 positive pairs, 6,234,466 negative pairs via 10% hash downsampling)
- **Features**: 19 identical E02 features computed on-the-fly via DuckDB in-memory engine
- **Hyperparameters**: Objective: binary, Boosting: GBDT, Learning Rate: 0.05, Num Leaves: 31, Seed: 2026, Num Rounds: 500, Threads: 2

### Training Runtimes:
* **Candidate Generation (Train + Test)**: 671.7s (~11.2 min)
* **Feature Extraction & Downsampling**: 1,207.0s (~20.1 min)
* **Fold 0 Training**: 274.2s (9,012,424 train rows, 4,019,908 pos)
* **Fold 1 Training**: 334.9s (9,002,317 train rows, 4,017,437 pos)
* **Fold 2 Training**: 283.8s (9,005,860 train rows, 4,017,150 pos)
* **Fold 3 Training**: 200.4s (9,011,332 train rows, 4,019,575 pos)
* **Fold 4 Training**: 233.4s (8,998,603 train rows, 4,018,602 pos)
* **Total 5-Fold Training**: **1,328.7s (~22.1 min)**

---

## 5. Full Canonical V2 OOF Integrity (Tasks 4 & 5)

Full out-of-fold scoring was executed across the complete 67,332,524 candidate pairs:

* **OOF Runtime**: 1,801.6s (~30.0 min)
* **Total Rows**: `67,332,524`
* **S2 Candidate Rows**: `30,359,040`
* **S3 Candidate Rows**: `36,973,484`
* **Duplicates**: `0`
* **Non-finite Scores**: `0`
* **Source-1 Entity Population Coverage**: `2,206,821 / 2,206,821` (100.0%)
* **Integrity Status**: **VERIFIED OK**

---

## 6. Performance Evaluation & Results (Task 6)

### Candidate Oracle Comparison:
| Metric | V1 Baseline | V2 Measured | Delta |
|---|---|---|---|
| **Overall Candidate Oracle** | `0.784522591` | **`0.827200935`** | **`+0.042678344`** |
| Fold 0 Oracle | `0.784651` | `0.827379` | `+0.042728` |
| Fold 1 Oracle | `0.784012` | `0.826718` | `+0.042706` |
| Fold 2 Oracle | `0.784890` | `0.827481` | `+0.042591` |
| Fold 3 Oracle | `0.785102` | `0.827720` | `+0.042618` |
| Fold 4 Oracle | `0.783960` | `0.826708` | `+0.042748` |

### Primary Model Score @ $T=0.50$ (Controlled Benchmark):
| Metric / Fold | V1 (E02 Baseline) | V2 Measured (E06) | Absolute Delta |
|---|---|---|---|
| **Overall Macro F0.5 @ $T=0.50$** | **`0.718765173`** | **`0.747084844`** | **`+0.028319671`** |
| Fold 0 @ $T=0.50$ | `0.718912` | `0.747122071` | `+0.028210` |
| Fold 1 @ $T=0.50$ | `0.718204` | `0.746511966` | `+0.028308` |
| Fold 2 @ $T=0.50$ | `0.719045` | `0.747279404` | `+0.028234` |
| Fold 3 @ $T=0.50$ | `0.719320` | `0.747798982` | `+0.028479` |
| Fold 4 @ $T=0.50$ | `0.718345` | `0.746713865` | `+0.028369` |

### Diagnostic Threshold Evaluations (Partial Available Sweep):
| Threshold | Macro F0.5 | Predicted Pairs | Status |
|---|---|---|---|
| **$T=0.500$** | **`0.747085`** | 5,883,557 | Completed (Primary) |
| **$T=0.600$** | **`0.755926`** | 5,659,564 | Completed (Diagnostic) |
| **$T=0.700$** | **`0.767933`** | 5,384,522 | Completed (Diagnostic) |
| **$T \in (0.70, 0.995]$** | *Stopped* | *Stopped* | Intentionally stopped per user request |

---

## 7. P3 Handoff & Validation Readiness

The canonical full out-of-fold prediction dataset is ready for independent P3 validation:

* **Artifact File**: `P2/reports/E06_V2_OOF_FULL_CANONICAL.tsv`
* **Size**: `2,928,257,277` bytes
* **Total Candidate Pairs**: `67,332,524`
* **Validation Target**: Verify Macro F0.5 @ $T=0.500$ equals `0.747084844` using `validation/scorer_v1.py` and confirm 5-fold distribution.
