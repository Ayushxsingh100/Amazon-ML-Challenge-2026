# Phase 4 to Phase 5 Handoff

**Handoff Scope**: Phase 4 Model Artifacts & Test Predictions  
**Recipient**: Phase 5 Submission & Post-Processing Pipeline  
**Date**: September 26, 2026  
**Status**: **VALIDATED ARTIFACTS AVAILABLE**

---

## 1. What Was Built

Phase 4 trained a 5-fold LightGBM ensemble on the frozen Phase 3 V3 candidates using 19 deterministic pairwise string and entity attribute features. The model was evaluated via leakage-safe S1-grouped cross-validation, calibrated across 9 decision thresholds, and used to score all 95,196,595 test candidate pairs.

---

## 2. Test Predictions Inventory

The primary handoff artifacts for Phase 5 are located in `P2/predictions/phase4/`:

1. **`P2/predictions/phase4/test_predictions_s2.tsv`**:
   - Rows: **43,841,928** (matches input test candidate count exactly)
   - Size: **1,510,836,431 bytes** (~1.41 GB)
   - SHA256: `1a4e94cc9b15ba984588db2b763e6c15967b2d7a7ef04213833d79dfe11307af`
   - Predicted Matches: **5,132,268** (11.7063%)

2. **`P2/predictions/phase4/test_predictions_s3.tsv`**:
   - Rows: **51,354,667** (matches input test candidate count exactly)
   - Size: **1,774,881,395 bytes** (~1.65 GB)
   - SHA256: `503282181cff0df65b86f3f19c2f2813b4cd84d99a1a2b4ed167ac2199d5e3aa`
   - Predicted Matches: **5,699,683** (11.0987%)

---

## 3. Prediction Schema & Meaning

Every row contains full traceability:
```tsv
source1_entity_id	candidate_entity_id	model_score	predicted_match_label
```
- `source1_entity_id`: Test S1 identifier (`S1-...`).
- `candidate_entity_id`: Test S2 or S3 identifier (`S2-...` or `S3-...`).
- `model_score`: Continuous ensemble predicted probability in `[0.0, 1.0]` (average of 5 fold models).
- `predicted_match_label`: Binary prediction (`1` = match, `0` = non-match) evaluated at the optimal threshold **$T = 0.60$**.
- Provenance:
  - Model: `LightGBM_5Fold_v1` (500 trees, lr=0.05, num_leaves=31)
  - Features: `v1_canonical_19` (19 pairwise deterministic features)
  - Target Sources: S2 evaluated in `test_predictions_s2.tsv`, S3 evaluated in `test_predictions_s3.tsv`

---

## 4. Key Performance Benchmarks

- **Candidate Retrieval Ceiling**: **72.161857%** (5,511,986 / 7,638,365 true pairs captured in V3).
- **Optimal Decision Threshold**: **$T = 0.60$**.
- **OOF Performance @ $T = 0.60$**:
  - Precision: **0.9795**
  - Recall: **0.9649**
  - Pairwise F0.5: **0.9765**
  - Macro F0.5 (Competition Metric): **0.841781**

---

## 5. Instructions for Phase 5

1. Phase 5 can directly group positive predictions (`predicted_match_label == 1`) by `source1_entity_id` to generate submission candidate lists.
2. Alternatively, Phase 5 may apply custom post-processing (e.g., top-k ranking per S1, graph-based transitive closure, or dynamic thresholding) using the raw continuous `model_score`.
