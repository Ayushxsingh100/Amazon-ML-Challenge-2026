# Phase 7 Reproducibility Manifest
## High-Recall Candidate Retrieval V4 & 5-Fold Grouped Validation

- **Date:** 2026-09-27
- **Base Git Commit:** `eae882c8be1a4f1e17ca9ad824b6b8fe657ba009`
- **Active Branch:** `phase7`
- **Platform:** macOS (Darwin 24.3.0, arm64)
- **Python Version:** `3.13.5` (Anaconda, Inc.)
- **DuckDB Version:** `1.5.5`
- **LightGBM Version:** `4.6.0`

---

## 1. Input Datasets & Checksums

| Dataset / File | File Path | Records / Rows | SHA256 Checksum |
| :--- | :--- | :--- | :--- |
| **Test S1 Parquet** | `P1/data/entities/test/source1/test_s1_entities.parquet` | 1,732,544 | Canonical entity set |
| **Test S2 Parquet** | `P1/data/entities/test/source2/test_s2_entities.parquet` | 4,887,273 | Canonical entity set |
| **Test S3 Parquet** | `P1/data/entities/test/source3/test_s3_entities.parquet` | 5,082,316 | Canonical entity set |
| **Train Ground Truth** | `outputs/person1_step1/train_ground_truth_reconstructed.tsv` | 2,206,821 | `70bc1d8a16c667e0155c2105d0ab2ebe41d7e7a85d8a529e3ca81c6c3a5af037` |
| **Fold Split Manifest** | `P3/reports/folds_v1_manifest.tsv` | 2,206,821 | 5-fold entity-grouped partition |
| **V3 S2 Test Predictions** | `P2/predictions/phase4/test_predictions_s2.tsv` | 43,841,928 | `1a4e94cc9b15ba984588db2b763e6c15967b2d7a7ef04213833d79dfe11307af` |
| **V3 S3 Test Predictions** | `P2/predictions/phase4/test_predictions_s3.tsv` | 51,354,667 | `503282181cff0df65b86f3f19c2f2813b4cd84d99a1a2b4ed167ac2199d5e3aa` |
| **Incremental V4 Predictions** | `P2/predictions/phase7/new_test_predictions_v4.parquet` | 1,850,216 | `3bf8b5a03e1e913a58e45447781da22e1bdf483b8d697858c2b5358aa09886b6` |

---

## 2. Preserved Baseline Checksums

| Artifact | File Path | Records / Rows | SHA256 Checksum | Leaderboard Score |
| :--- | :--- | :--- | :--- | :--- |
| **Phase 6 Baseline** | `P2/predictions/phase7/baseline_matching_results.tsv` | 1,732,544 | `fce26bfc78d6c2a74829ff512981d94d3b66d89ca5e1d232b14ea1eafa402f9a` | **~0.764** |
| **Phase 5 Baseline** | `P2/predictions/phase6/baseline_matching_results.tsv` | 1,732,544 | `c0d5da59dfafabd771ca5fbd1dba655b41fff458af9ff470e3a2ecb86535bbbc` | **~0.702** |

---

## 3. Output Submission Artifacts

| Artifact | File Path | File Size (Bytes) | SHA256 Checksum |
| :--- | :--- | :--- | :--- |
| **New Phase 7 Submission** | `output/matching_results.tsv` | 83,577,391 | `02c66b52ef31b10ed16f68ba0592edd70015a94c3679f139987d8dfba2741c98` |
| **Mirrored Phase 7 Copy** | `P2/predictions/phase7/matching_results.tsv` | 83,577,391 | `02c66b52ef31b10ed16f68ba0592edd70015a94c3679f139987d8dfba2741c98` |

---

## 4. Pipeline Parameters

- **Candidate Generator:** V4 (V3 Baseline + Strategy A Address Clean 15 + Strategy B Name Clean Core No House)
- **Model:** 5-Fold LightGBM ensemble (`lgb_fold0` through `lgb_fold4`)
- **Threshold:** $T = 0.88$
- **Score Margin:** $\Delta \le 0.05$
- **Target Cap:** $K \le 8$
- **Total Test Candidates:** 97,046,811 (95,196,595 V3 + 1,850,216 incremental V4)
- **Total Test Predictions:** 4,738,677 matched pairs
- **Total Test S1 Entities Matched:** 1,562,604 (90.19%)
- **Total Test Singletons:** 169,940 (9.81%)

---

## 5. Execution Pipeline Commands

```bash
# Step 1: Run missed GT forensics
python3 P1/scripts/phase7_missed_gt_analysis.py

# Step 2: Run candidate blocking ablation
python3 P1/scripts/phase7_blocking_ablation.py

# Step 3: Run 5-fold grouped OOF evaluation
python3 P2/scripts/phase7_evaluate_all_folds.py

# Step 4: Score incremental V4 test candidates
python3 P2/scripts/phase7_score_new_test_candidates.py

# Step 5: Generate and validate Phase 7 final submission
python3 P2/scripts/phase7_generate_submission.py

# Step 6: Verify checksum
shasum -a 256 output/matching_results.tsv
# Expected: 02c66b52ef31b10ed16f68ba0592edd70015a94c3679f139987d8dfba2741c98
```
