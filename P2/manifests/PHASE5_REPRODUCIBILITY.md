# Phase 5 Reproducibility Manifest

## 1. Environment & Version Control
- **Repository**: `https://github.com/Ayushxsingh100/Amazon-ML-Challenge-2026.git`
- **Current Git Branch**: `main`
- **Base Commit**: `f21c81dc2918986ab5393e7d1a3922e8a3caa3ee` (Ahead of `origin/main` by 2 commits)
- **Execution Platform**: macOS (Darwin 24.3.0, arm64)
- **Python Version**: `3.13.5` (Anaconda, Inc.)
- **DuckDB Version**: `1.5.5`

---

## 2. Input Datasets & Artifacts

| Category | File Path | Records / Rows | SHA256 Checksum |
| :--- | :--- | :--- | :--- |
| **S2 Predictions** | `P2/predictions/phase4/test_predictions_s2.tsv` | 43,841,928 | `1a4e94cc9b15ba984588db2b763e6c15967b2d7a7ef04213833d79dfe11307af` |
| **S3 Predictions** | `P2/predictions/phase4/test_predictions_s3.tsv` | 51,354,667 | `503282181cff0df65b86f3f19c2f2813b4cd84d99a1a2b4ed167ac2199d5e3aa` |
| **Test S1 Entities** | `P1/data/entities/test/source1/test_s1_entities.parquet` | 1,732,544 | Canonical entity set |
| **Test S2 Entities** | `P1/data/entities/test/source2/test_s2_entities.parquet` | 4,887,273 | Canonical entity set |
| **Test S3 Entities** | `P1/data/entities/test/source3/test_s3_entities.parquet` | 5,082,316 | Canonical entity set |
| **Train Ground Truth** | `outputs/person1_step1/train_ground_truth_reconstructed.tsv` | 2,206,821 | `70bc1d8a16c667e0155c2105d0ab2ebe41d7e7a85d8a529e3ca81c6c3a5af037` |

---

## 3. Official Scorer & Evaluation Framework
- **Official Scorer File**: `validation/scorer_v1.py`
- **Supporting Metrics**: `validation/metrics.py`
- **Scoring Function**: `score_predictions(predictions, ground_truth, entity_ids)`
- **Target Metric**: Entity-level Macro $F_{0.5}$ (Precision weighted at $\beta = 0.5$).

---

## 4. Historical Phase 4 Validation Artifacts
- **OOF Sweep Report**: `P2/reports/phase4_oof_sweep_results.json`
- **Modeling Metrics Report**: `P2/reports/phase4_modeling_metrics.json`
- **Trained Boosters**: `P2/models/phase4/lgb_fold[0-4].txt`
- **Cross-Validation Split**: `P3/reports/folds_v1_manifest.tsv`
- **Historical OOF Score**: Peak Macro $F_{0.5} = 0.841781$ at threshold $T = 0.60$.

---

## 5. Artifact Limitations & Constraints
- **Purged Pair-Level OOF Predictions**: The raw pair-level OOF prediction tables were deleted during Phase 4 training to conserve disk space. The 9-point threshold sweep summary is fully documented and preserved in `phase4_oof_sweep_results.json`.
- **Test Ground Truth**: Competition test ground truth is held out. Supervised metrics on the test split are not computable.
- **Graph / Transitive Closure**: Not used. Pairwise S1 $\rightarrow$ {S2, S3} predictions are directly serialized without transitive clustering to protect precision.

---

## 6. Generation Parameters & Policy
- **Selected Threshold**: $T = 0.60$
- **Selection Policy**: Retain all candidate pairs with `predicted_match_label == 1` (`model_score >= 0.60`).
- **Multiple Matches**: Retained without top-1 reduction.
- **Candidate Ordering**: `model_score DESC, candidate_entity_id ASC`.
- **Entity Ordering**: `source1_entity_id ASC`.
- **No-Match Format**: Empty string `""` with zero characters after the tab (`S1-xxxxx\t\n`).
- **Generation Script**: `P2/scripts/phase5_generate_submission.py`

---

## 7. Execution & Reproduction Command

To reproduce the submission exactly:
```bash
python3 P2/scripts/phase5_generate_submission.py
```

---

## 8. Final Submission Artifacts

| Property | Value |
| :--- | :--- |
| **Primary Submission Path** | `P2/predictions/phase5/matching_results.tsv` |
| **Mirrored Submission Path** | `output/matching_results.tsv` |
| **File Format** | TSV (`\t` delimiter, UTF-8, no quotes) |
| **Header** | `source1_entity_id\tmatched_entity_ids` |
| **Data Rows** | 1,732,544 |
| **Total Lines** | 1,732,545 |
| **File Size** | 162,077,141 bytes (~154.57 MiB) |
| **SHA256 Checksum** | `c0d5da59dfafabd771ca5fbd1dba655b41fff458af9ff470e3a2ecb86535bbbc` |
| **Matched Entities** | 1,600,875 (92.40%) |
| **No-Match Entities** | 131,669 (7.60%) |
| **Total Matched Pairs** | 10,831,951 (S2: 5,132,268; S3: 5,699,683) |
| **Distinct Target Entities** | 4,754,500 |
| **Validation Status** | **PASS WITH WARNINGS** |
