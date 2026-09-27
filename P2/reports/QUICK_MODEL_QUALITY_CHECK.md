# Quick Model Quality Check

## 1. Existing OOF Validation

The authoritative entity-resolution modeling evaluation is recorded from the 5-fold S1-grouped cross-validation performed in Phase 4. The model architecture is a 5-fold LightGBM ensemble trained on the 19 canonical pairwise features, evaluating candidate pairs generated during Phase 3.

At the selected decision threshold **$T = 0.60$**, the recorded out-of-fold (OOF) performance metrics are:
- **Pairwise Precision**: `0.979470` (97.95%)
- **Pairwise Recall**: `0.964903` (96.49%)
- **Pairwise $F_{0.5}$**: `0.976527`
- **Recorded 5-fold S1-grouped OOF Macro $F_{0.5}$**: `0.841781`
- **Predicted Positives across OOF validation**: `5,430,010` (out of 5,511,986 ground-truth pairs in candidate set)

---

## 2. Threshold Sweep

The complete out-of-fold decision threshold sweep across thresholds 0.10 through 0.90, recorded in [P2/reports/phase4_oof_sweep_results.json](file:///Users/krishnagera/Amazon-ML-Challenge-2026/P2/reports/phase4_oof_sweep_results.json), is as follows:

| Threshold | True Positives (TP) | False Positives (FP) | False Negatives (FN) | True Negatives (TN) | Pairwise Precision | Pairwise Recall | Pairwise $F_{0.5}$ | Macro $F_{0.5}$ | Predicted Positives |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.10** | 5,466,872 | 456,710 | 45,114 | 8,313,860 | 0.922899 | 0.991815 | 0.935817 | 0.824318 | 5,923,582 |
| **0.20** | 5,439,739 | 304,961 | 72,247 | 8,465,609 | 0.946914 | 0.986893 | 0.954631 | 0.833719 | 5,744,700 |
| **0.30** | 5,413,348 | 228,301 | 98,638 | 8,542,269 | 0.959533 | 0.982105 | 0.963952 | 0.838145 | 5,641,649 |
| **0.40** | 5,387,498 | 180,883 | 124,488 | 8,589,687 | 0.967516 | 0.977415 | 0.969476 | 0.840382 | 5,568,381 |
| **0.50** | 5,356,393 | 143,723 | 155,593 | 8,626,847 | 0.973869 | 0.971772 | 0.973449 | 0.841605 | 5,500,116 |
| **0.60** | **5,318,532** | **111,478** | **193,454** | **8,659,092** | **0.979470** | **0.964903** | **0.976527** | **0.841781** | **5,430,010** |
| **0.70** | 5,257,121 | 78,259 | 254,865 | 8,692,311 | 0.985332 | 0.953762 | 0.978878 | 0.840502 | 5,335,380 |
| **0.80** | 5,168,058 | 49,371 | 343,928 | 8,721,199 | 0.990537 | 0.937604 | 0.979435 | 0.836630 | 5,217,429 |
| **0.90** | 5,039,324 | 27,754 | 472,662 | 8,742,816 | 0.994523 | 0.914248 | 0.977531 | 0.828438 | 5,067,078 |

*Observation*: Peak Macro $F_{0.5}$ is achieved at $T = 0.60$ (`0.841781`).

---

## 3. Raw OOF Artifact Availability

An exhaustive scan of the repository was conducted for raw pair-level OOF tables:
- **Search Paths**: `P2/data/`, `P2/reports/`, `outputs/`
- **Result**: Raw pair-level OOF predictions are not available; the recorded Phase4 OOF metrics are the authoritative validation result.
- **Context**: The raw candidate-level OOF tables (~93.1M candidate rows across 5 folds) were purged during Phase 4 training to conserve disk space, while retaining the full threshold sweep metrics in `phase4_oof_sweep_results.json` and `phase4_modeling_metrics.json`.

---

## 4. Phase4 Test Prediction Integrity

Both Phase 4 test prediction files were audited row-by-row using DuckDB:

| Attribute | `P2/predictions/phase4/test_predictions_s2.tsv` | `P2/predictions/phase4/test_predictions_s3.tsv` | Combined Status |
| :--- | :--- | :--- | :--- |
| **Row Count** | 43,841,928 | 51,354,667 | Exact match (95,196,595 total rows) |
| **Null S1 IDs** | 0 | 0 | Zero nulls |
| **Null Candidate IDs** | 0 | 0 | Zero nulls |
| **Duplicate Pairs** | 0 | 0 | All pairs unique per file |
| **Score Range** | `[0.0, 1.0]` | `[0.0, 1.0]` | Fully valid continuous probabilities |
| **Labels Set** | `{0, 1}` | `{0, 1}` | Strictly binary |
| **Threshold Alignment** | 100% of rows with `label == 1` have `score >= 0.60`. (244 S2 and 214 S3 rows with unrounded prob $< 0.60$ but rounded to 0.6000 correctly retained `label == 0`). | 100% of rows with `label == 1` have `score >= 0.60`. (214 rows with unrounded prob $< 0.60$ rounded to 0.6000 retained `label == 0`). | Verified numerical precision behavior |
| **Target Separation** | 100% prefix `S2-` (0 S3, 0 S1) | 100% prefix `S3-` (0 S2, 0 S1) | Completely disjoint & clean |

---

## 5. Final Submission Integrity

The production submission file [output/matching_results.tsv](file:///Users/krishnagera/Amazon-ML-Challenge-2026/output/matching_results.tsv) was validated against canonical entities and Phase 4 predictions:

- **Row Count**: Exactly 1,732,544 data rows (+1 header row = 1,732,545 lines).
- **Test S1 Universe**: Exactly 1,732,544 unique S1 IDs matching [test_s1_entities.parquet](file:///Users/krishnagera/Amazon-ML-Challenge-2026/P1/data/entities/test/source1/test_s1_entities.parquet) (0 missing, 0 unexpected, 0 duplicate S1 rows).
- **Target Domain Validity**: All 10,831,951 matched target instances exist in [test_s2_entities.parquet](file:///Users/krishnagera/Amazon-ML-Challenge-2026/P1/data/entities/test/source2/test_s2_entities.parquet) (5,132,268) or [test_s3_entities.parquet](file:///Users/krishnagera/Amazon-ML-Challenge-2026/P1/data/entities/test/source3/test_s3_entities.parquet) (5,699,683).
- **Contamination**: 0 training target IDs, 0 malformed target IDs, 0 S1 target IDs.
- **Target Uniqueness within S1**: 0 duplicate target IDs inside any S1 candidate list.
- **Traceability to Phase 4**: Exactly 10,831,951 submitted pairs match the 10,831,951 Phase 4 positive predictions (`model_score >= 0.60` and `predicted_match_label == 1`).
  - Dropped Phase 4 positive predictions: **0**
  - Unpredicted extra pairs introduced: **0**
- **No-Match Formatting**: Exactly 131,669 rows (7.60%) contain an empty string after the tab (`S1-xxxxx\t\n`). Zero forbidden tokens (`NULL`, `None`, `NA`, `[]`, `{}`, `NO_MATCH`, or `""`).

---

## 6. Exact Metrics

A clear distinction is maintained across the pipeline stages:

1. **Candidate-Generation Recall (Phase 3 V3)**:
   - S2 Candidate Recall: `72.481785%`
   - S3 Candidate Recall: `71.862295%`
   - **Combined Candidate Recall**: `72.161857%` (on 7,638,365 ground-truth pairs)
2. **Model Out-of-Fold Performance (Phase 4)**:
   - **Recorded 5-fold S1-grouped OOF Macro $F_{0.5}$ at threshold 0.60 = 0.841781**
   - Pairwise Precision: `0.979470`
   - Pairwise Recall: `0.964903`
   - Pairwise $F_{0.5}$: `0.976527`
3. **Final Test Prediction Statistics (Phase 5)**:
   - Total Scored Test Candidates: `95,196,595`
   - Submitted Positive Pairs: `10,831,951` (Positive Rate: 11.38%)
   - Matched S1 Entities: `1,600,875` (92.40%)
   - Unmatched S1 Entities: `131,669` (7.60%)
   - Single-Match S1 Entities: `241,057` (13.91%)
   - Multi-Match S1 Entities: `1,359,818` (78.49%)
4. **Final Submission Integrity**:
   - Header: `source1_entity_id\tmatched_entity_ids`
   - Line Count: `1,732,545`
   - SHA256: `c0d5da59dfafabd771ca5fbd1dba655b41fff458af9ff470e3a2ecb86535bbbc`

The actual test-set accuracy cannot be measured without test ground truth.

---

## 7. Limitations

1. **Unobserved Test Ground Truth**: There is no test ground truth available in the repository. All test-stage evaluations represent structural integrity, schema conformity, coverage analysis, and consistency checks against Phase 4 predictions.
2. **Purged Raw OOF Tables**: The raw candidate-level OOF predictions were purged during Phase 4 training to conserve disk space. The recorded Phase 4 OOF threshold sweep serves as the authoritative validation record.
3. **Retrieval Bound**: Downstream matching recall is inherently bounded by the Phase 3 candidate retrieval recall (72.16% on training data).

---

## 8. Final Status

**PASS WITH WARNINGS**

- **Justification**: All test predictions, canonical entity domains, schema formats, and threshold consistencies passed audit with 100% data integrity. The warning reflects the non-blocking limitation that raw candidate-level OOF tables were purged during Phase 4 training to conserve disk space.
