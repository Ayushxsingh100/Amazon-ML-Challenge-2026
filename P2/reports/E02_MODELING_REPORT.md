# E02 Canonical Modeling Baseline Report

## Summary
The canonical LightGBM E02 root baseline has successfully completed training on the frozen `cands_BCD_v1` candidate universe and generated the official out-of-fold (OOF) predictions across the synchronized P3 5-fold validation manifest.

### Final Baseline Decision
- **Official Threshold**: Pre-registered fixed `T=0.5`
- **Official P3 Scorer Result**: **`0.7695`** (Macro F0.5)

## 1. Environment & State
- **Exact commands run**: 
  - `python -u P2/scripts/e02_train_lgb.py` (which invoked DuckDB arrow extraction, LGBM cv, and validation sweep).
- **Final Git SHA**: `60a8658579cfaca0e9e22ba67660a4dc65dc1832`
- **BCD Implementation SHA**: `76316c7`
- **Runtime**: ~33 minutes (1994.29 seconds)
- **Memory**: Max memory limited via 10% deterministic downsampling of negatives to ensure LightGBM runs within typical desktop memory bounds (< 5 GB).

## 2. Dataset & Sampling
- **Total Canonical Candidates (S2+S3)**: 54,592,725 pairs (4,560,579 Positives, 50,032,146 Negatives)
- **Sampling Rate**: `10%` deterministic hash-based sampling of the negative class only.
- **Training Row Count**: `9,566,660` exact retained rows.
- **Positives Retained**: `4,560,579` (100% of all positives).
- **Negatives Retained**: `5,006,081`.
- **Fold Coverage**: Exactly matched the 2,206,821 unique S1 entities from `P3/reports/folds_v1_manifest.tsv`.

## 3. LightGBM Parameters
Using the pre-registered canonical configuration:
- `objective` = 'binary'
- `boosting_type` = 'gbdt'
- `learning_rate` = 0.05
- `num_leaves` = 31
- `max_depth` = -1
- `feature_fraction` = 1.0
- `bagging_fraction` = 1.0
- `min_data_in_leaf` = 20
- `lambda_l1` = 0.0
- `lambda_l2` = 0.0
- `seed` = 2026
- `num_boost_round` = 500

## 4. Evaluation & Diagnostics

### OOF Coverage
- **Total Predictions**: `9,566,660` rows predicted.
- **Integrity Checks**: `0` duplicate predictions, `0` missing predictions, `True` all scores finite.

### Threshold Sweep Summary (Diagnostics)
| Threshold | Macro F0.5 |
|-----------|------------|
| 0.1       | 0.7629     |
| 0.2       | 0.7673     |
| 0.3       | 0.7688     |
| 0.4       | 0.7693     |
| **0.5 (Base)**| **0.7695** |
| 0.6       | 0.7691     |
| 0.7       | 0.7677     |
| 0.8       | 0.7648     |
| 0.9       | 0.7593     |

### Oracle / Loss Decomposition
- **Candidate Floor (Predict empty for all S1)**: `0.0558` (Reflects S1 entities where Ground Truth is genuinely empty).
- **Canonical E02 Baseline**: `0.7695`
- **Candidate Oracle (Perfect thresholding/ranking over captured positives)**: `0.7845`
  
*Analysis*: The baseline model achieves 0.7695 out of a theoretical maximum of 0.7845 reachable on this candidate universe. The pairwise ranking/classification loss is therefore ~0.015, while the primary remaining loss stems from candidate generation (recall ceiling) and true missing information.

## 5. Major Error Clusters
Generated `P2/reports/E02_ERROR_ANALYSIS.tsv` contains:
- **High-confidence False Positives**: Usually short/ambiguous names with matching numerical address prefixes that were negative in ground truth.
- **Low-confidence True Positives**: Substantial edit distance or missing tokens in S2/S3 features that LightGBM penalized.
- **False Negatives**: Pairs assigned low scores that actually were correct matches (heavy reliance on name_len_diff/address_exact_match).

## 6. Output Paths & P3 Handoff Artifacts
All outputs have been safely saved to the frozen P2 workspace:
1. `P2/reports/E02_MODELING_REPORT.md` (This document)
2. `P2/reports/E02_RUN.json` (Full execution metadata and metric breakdown)
3. `P2/reports/E02_THRESHOLD_SWEEP.tsv` (Diagnostic grid evaluation)
4. `P2/reports/E02_ERROR_ANALYSIS.tsv` (Top extreme false positives/negatives)
5. `P2/reports/E02_OOF_MANIFEST.tsv` (The canonical 9.5M out-of-fold prediction probabilities)
6. `P2/models/e02_lgb_fold{0-4}.txt` (Saved LightGBM models)

These artifacts exactly fulfill the P3 protocol for the authoritative scorer handoff. No further tuning is required. E02 is completed.
