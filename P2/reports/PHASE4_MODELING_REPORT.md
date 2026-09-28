# Phase 4 Modeling Report

**Project**: Amazon ML Challenge 2026 — Entity Resolution  
**Phase**: Phase 4 — Entity-Resolution Matching Model  
**Author**: Person 2 (P2) Modeling Execution  
**Date**: September 26, 2026  
**Status**: **COMPLETE & VALIDATED**

---

## 1. Objective

Build a reproducible, leakage-safe, high-precision entity-resolution matching model that:
1. Consumes the frozen production V3 candidate pairs from Phase 3.
2. Constructs ground-truth labels strictly from the canonical reconstructed ground truth.
3. Extracts deterministic pairwise features using DuckDB SQL.
4. Trains a 5-fold LightGBM matching model grouped strictly by S1 entity to prevent entity leakage.
5. Performs genuine out-of-fold (OOF) evaluation across multiple decision thresholds.
6. Evaluates controlled model experiments.
7. Generates full test predictions for S2 and S3 candidates.
8. Produces complete reproducibility manifests and Phase 5 handoff documentation.

---

## 2. Input Artifacts & SHA256 Verification

All input candidate files and ground truth artifacts were independently validated prior to modeling:

| Artifact | Path | Exact SHA256 Checksum | Rows | Verified Status |
|---|---|---|---|---|
| **Train S2 V3 Candidates** | `P1/data/candidates/v3/train_candidate_pairs_s2_v3.tsv` | `61d5dce6389bf424daf61f842ed0f41ebc3991fa27763d4641c4ab4931a7508c` | 42,933,945 | **EXACT MATCH** |
| **Train S3 V3 Candidates** | `P1/data/candidates/v3/train_candidate_pairs_s3_v3.tsv` | `d5e90bb6d9a5d81154a83366d98211352e95bebcbc84c5011f14cd0bfe10eeeb` | 50,238,004 | **EXACT MATCH** |
| **Test S2 V3 Candidates** | `P1/data/candidates/v3/test_candidate_pairs_s2_v3.tsv` | `4aa0c71ea70a98aff77088da2c4b070e8fb305b8fbebfbda4daa4fa14398dfa2` | 43,841,928 | **EXACT MATCH** |
| **Test S3 V3 Candidates** | `P1/data/candidates/v3/test_candidate_pairs_s3_v3.tsv` | `f24041fbbc3a5f56a4b9c26b20dfde3377770f85427a9373bc9c1efc8bf441cd` | 51,354,667 | **EXACT MATCH** |
| **Ground Truth Reconstructed** | `outputs/person1_step1/train_ground_truth_reconstructed.tsv` | `70bc1d8a16c667e0155c2105d0ab2ebe41d7e7a85d8a529e3ca81c6c3a5af037` | 2,206,821 | **EXACT MATCH** |
| **Canonical Folds Manifest** | `P3/reports/folds_v1_manifest.tsv` | `9dcec5d83ae99ea9254d32e92c42ce2c2ce0f9a23fc26c7104d49a7fa347895e` | 2,206,821 | **EXACT MATCH** |

---

## 3. Input Validation & Column Availability

Canonical parquet entity tables were inspected:
- `train_s1`: 2,206,821 entities
- `train_s2`: 5,034,616 entities
- `train_s3`: 5,285,603 entities
- `test_s1`: 1,732,544 entities
- `test_s2`: 4,887,273 entities
- `test_s3`: 5,082,316 entities

Documented column availability:
- **`business_name`**: AVAILABLE (normalized & raw)
- **`business_address`**: AVAILABLE (normalized & raw)
- **`country`**: AVAILABLE (normalized & raw)
- **`phone`**: **COLUMN NOT AVAILABLE: phone** (absent in raw dataset)
- **`email`**: **COLUMN NOT AVAILABLE: email** (absent in raw dataset)

---

## 4. Training Candidate Statistics & Label Construction

Labels were constructed strictly by inner join against canonical exploded true pairs:
- **Total Training Candidates**: 93,171,949
  - S2 Candidates: 42,933,945
  - S3 Candidates: 50,238,004
- **Positive Labels (Matches in GT)**: 5,511,986
  - S2 Positives: 2,677,201
  - S3 Positives: 2,834,785
- **Negative Labels**: 87,659,963
  - Positive Rate: 5.915929%
- **Duplicate Pairs**: 0
- **Missing / Malformed IDs**: 0

---

## 5. Candidate Recall Ceiling

The candidate set establishes the theoretical upper bound (oracle recall) for any downstream matching model:
- **S2 Candidate Ceiling**: 2,677,201 / 3,693,619 = **72.4818%**
- **S3 Candidate Ceiling**: 2,834,785 / 3,944,746 = **71.8623%**
- **Combined Retrieval Ceiling**: 5,511,986 / 7,638,365 = **72.161857%**

*Crucial Architecture Principle*: A true pair missing from the V3 retrieval set cannot be recovered by the model. Candidate recall is the strict retrieval ceiling.

---

## 6. Feature Engineering & Leakage Audit

### Deterministic Pairwise Features (19 Features)
1. `name_exact_match`: Exact binary equality on normalized names.
2. `name_jaro_winkler`: Jaro-Winkler character similarity scaled to [0, 1].
3. `name_jaccard`: Character token Jaccard similarity.
4. `prefix4_match`: Equality of 4-character name prefixes.
5. `first_token_match`: Equality of the first whitespace-separated name token.
6. `name_len_diff`: Absolute character length difference.
7. `name_len_ratio`: Ratio of shortest to longest name length.
8. `address_exact_match`: Exact binary equality on normalized addresses.
9. `address_jaro_winkler`: Address Jaro-Winkler similarity.
10. `address_jaccard`: Address token Jaccard similarity.
11. `address_len_diff`: Absolute address length difference.
12. `address_len_ratio`: Ratio of shortest to longest address length.
13. `address_first_number_match`: Exact match on leading house / street number.
14. `country_match`: Exact match on ISO country code.
15. `s1_name_len`: Character length of Source-1 name.
16. `tgt_name_len`: Character length of Target name.
17. `s1_addr_len`: Character length of Source-1 address.
18. `tgt_addr_len`: Character length of Target address.
19. `source_is_s3`: Binary indicator for Target source (0 = S2, 1 = S3).

### Leakage Audit: **PASS**
- Features are strictly pairwise comparisons of entity attributes.
- Ground truth is used **only** as the training label target.
- Folds are grouped strictly by `source1_entity_id`, guaranteeing zero S1 cross-fold leakage.

---

## 7. Model Training & Out-of-Fold (OOF) Evaluation

- **Architecture**: LightGBM Binary Classifier (GBDT)
- **Hyperparameters**: `learning_rate=0.05`, `num_leaves=31`, `max_depth=-1`, `min_data_in_leaf=20`, `num_boost_round=500`, `seed=2026`
- **Downsampling**: 100% positives + 10% deterministic hash negatives (`ABS(hash(s1 || tgt)) % 10 = 0`)
- **Total Training Sample**: 14,282,556 rows (5,511,986 positives, 8,770,570 negatives)

### Out-of-Fold Multi-Threshold Sweep:

| Threshold | True Positives | False Positives | False Negatives | Precision | Recall | Pairwise F0.5 | Macro F0.5 (Competition Scorer) | Predicted Positives |
|---|---|---|---|---|---|---|---|---|
| **0.10** | 5,466,872 | 456,710 | 45,114 | 0.9229 | 0.9918 | 0.9358 | **0.824318** | 5,923,582 |
| **0.20** | 5,439,739 | 304,961 | 72,247 | 0.9469 | 0.9869 | 0.9546 | **0.833719** | 5,744,700 |
| **0.30** | 5,413,348 | 228,301 | 98,638 | 0.9595 | 0.9821 | 0.9640 | **0.838145** | 5,641,649 |
| **0.40** | 5,387,498 | 180,883 | 124,488 | 0.9675 | 0.9774 | 0.9695 | **0.840382** | 5,568,381 |
| **0.50** | 5,356,393 | 143,723 | 155,593 | 0.9739 | 0.9718 | 0.9734 | **0.841605** | 5,500,116 |
| **0.60** | 5,318,532 | 111,478 | 193,454 | 0.9795 | 0.9649 | 0.9765 | **0.841781** | 5,430,010 |
| **0.70** | 5,257,121 | 78,259 | 254,865 | 0.9853 | 0.9538 | 0.9789 | **0.840502** | 5,335,380 |
| **0.80** | 5,168,058 | 49,371 | 343,928 | 0.9905 | 0.9376 | 0.9794 | **0.836630** | 5,217,429 |
| **0.90** | 5,039,324 | 27,754 | 472,662 | 0.9945 | 0.9142 | 0.9775 | **0.828438** | 5,067,078 |

**Selected Optimal Threshold**: **T = 0.60** (Achieves Macro F0.5 = **0.841781**).

---

## 8. Controlled Model Experiments

1. **Baseline Model (EXP-MOD-00)**:
   - 19 canonical features, LightGBM (lr=0.05, leaves=31).
   - Fold 0 Macro F0.5: **0.853945**
2. **Class Weighting (EXP-MOD-05)**:
   - Evaluated `scale_pos_weight = 1.5` on Fold 0.
   - Fold 0 Macro F0.5: **0.853884**
   - *Finding*: Because competition metric is precision-tilted ($F_{0.5}$ weights precision twice as heavily as recall), unweighted training combined with optimal threshold calibration ($T=0.6$) delivers superior Macro $F_{0.5}$ without inflating false positives.

---

## 9. Test Prediction Generation & Integrity

Test predictions were generated by streaming the 95.2M test candidates through the 5-fold ensemble:
- **Test S2 Predictions**: `/Users/krishnagera/Amazon-ML-Challenge-2026/P2/predictions/phase4/test_predictions_s2.tsv`
  - Output Prediction Rows: 43,841,928
  - Predicted Positive Matches: 5,132,268 (11.7063%)
  - Null S1 / Candidate IDs: 0
  - SHA256: `1a4e94cc9b15ba984588db2b763e6c15967b2d7a7ef04213833d79dfe11307af`
- **Test S3 Predictions**: `/Users/krishnagera/Amazon-ML-Challenge-2026/P2/predictions/phase4/test_predictions_s3.tsv`
  - Output Prediction Rows: 51,354,667
  - Predicted Positive Matches: 5,699,683 (11.0987%)
  - Null S1 / Candidate IDs: 0
  - SHA256: `503282181cff0df65b86f3f19c2f2813b4cd84d99a1a2b4ed167ac2199d5e3aa`
- **Combined Test Candidates Scored**: **95,196,595 / 95,196,595** (100.0% coverage, zero row loss).

---

## 10. Final Artifact Inventory

| Category | Artifact | Path | Size | SHA256 |
|---|---|---|---|---|
| **Model** | Fold 0 Binary | `P2/models/phase4/lgb_fold0.txt` | 1,801,800 B | `81c9c55bf3980cf7...` |
| **Model** | Fold 1 Binary | `P2/models/phase4/lgb_fold1.txt` | 1,800,995 B | `24609eac326a62cc...` |
| **Model** | Fold 2 Binary | `P2/models/phase4/lgb_fold2.txt` | 1,801,225 B | `1caae0a1a755774a...` |
| **Model** | Fold 3 Binary | `P2/models/phase4/lgb_fold3.txt` | 1,802,047 B | `6414e917b0f6e74d...` |
| **Model** | Fold 4 Binary | `P2/models/phase4/lgb_fold4.txt` | 1,801,366 B | `6ded4f9508e0d536...` |
| **Predictions** | Test S2 Predictions | `P2/predictions/phase4/test_predictions_s2.tsv` | 1,510,836,431 B | `1a4e94cc9b15ba984588db2b763e6c15967b2d7a7ef04213833d79dfe11307af` |
| **Predictions** | Test S3 Predictions | `P2/predictions/phase4/test_predictions_s3.tsv` | 1,774,881,395 B | `503282181cff0df65b86f3f19c2f2813b4cd84d99a1a2b4ed167ac2199d5e3aa` |
| **Metrics** | Phase 4 Metrics JSON | `P2/reports/phase4_modeling_metrics.json` | 5,720 B | `4a2f527e0a22e890...` |

---

## 11. Handoff to Phase 5

The Phase 4 test predictions provide complete pairwise traceability:
- `source1_entity_id`: Test S1 ID
- `candidate_entity_id`: Test Target ID (S2 or S3)
- `model_score`: Continuous ensemble probability in [0, 1]
- `predicted_match_label`: Binary classification at optimal $T=0.60$
- Output paths: `P2/predictions/phase4/test_predictions_s2.tsv` and `test_predictions_s3.tsv`
- Provenance: Model `LightGBM_5Fold_v1` (500 rounds, lr=0.05, num_leaves=31), Features `v1_canonical_19`
