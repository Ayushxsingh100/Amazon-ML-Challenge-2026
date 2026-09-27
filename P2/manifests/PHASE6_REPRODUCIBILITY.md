# Phase 6 Reproducibility Manifest
## Calibrated Entity Resolution Pipeline & Leaderboard Improvement

- **Date:** 2026-09-27
- **Author/Role:** Lead ML/Entity-Resolution Engineer
- **Repository:** `https://github.com/Ayushxsingh100/Amazon-ML-Challenge-2026.git`
- **Execution Platform:** macOS (Darwin 24.3.0, arm64)
- **Python Version:** `3.13.5` (Anaconda, Inc.)
- **DuckDB Version:** `1.5.5`

---

## 1. Input Datasets & Checksums

| Category | File Path | Records / Rows | SHA256 Checksum |
| :--- | :--- | :--- | :--- |
| **S2 Test Predictions** | `P2/predictions/phase4/test_predictions_s2.tsv` | 43,841,928 | `1a4e94cc9b15ba984588db2b763e6c15967b2d7a7ef04213833d79dfe11307af` |
| **S3 Test Predictions** | `P2/predictions/phase4/test_predictions_s3.tsv` | 51,354,667 | `503282181cff0df65b86f3f19c2f2813b4cd84d99a1a2b4ed167ac2199d5e3aa` |
| **Test S1 Entities** | `P1/data/entities/test/source1/test_s1_entities.parquet` | 1,732,544 | Canonical entity set |
| **Test S2 Entities** | `P1/data/entities/test/source2/test_s2_entities.parquet` | 4,887,273 | Canonical entity set |
| **Test S3 Entities** | `P1/data/entities/test/source3/test_s3_entities.parquet` | 5,082,316 | Canonical entity set |
| **Train Ground Truth** | `outputs/person1_step1/train_ground_truth_reconstructed.tsv` | 2,206,821 | `70bc1d8a16c667e0155c2105d0ab2ebe41d7e7a85d8a529e3ca81c6c3a5af037` |
| **Phase 4 Boosters** | `P2/models/phase4/lgb_fold[0-4].txt` | 5 models | 5-fold cross-validation LightGBM boosters |
| **Fold Manifest** | `P3/reports/folds_v1_manifest.tsv` | 2,206,821 | 5-fold grouped entity split |

---

## 2. Preserved Baseline Submission

| Property | Value |
| :--- | :--- |
| **Baseline Path** | `P2/predictions/phase6/baseline_matching_results.tsv` |
| **File Size** | 162,077,141 bytes |
| **SHA256 Checksum** | `c0d5da59dfafabd771ca5fbd1dba655b41fff458af9ff470e3a2ecb86535bbbc` |
| **Policy** | Phase 5 uncalibrated threshold ($T=0.60$) with zero margin/rank filtering |
| **Public Leaderboard Score** | $\sim 0.702$ |

---

## 3. Decision Policy Parameters

The Phase 6 calibrated submission enforces:
1. **Probability Cutoff:** $T = 0.86$
2. **Per-S1 Score Margin:** $(\text{top1\_score} - \text{candidate\_score}) \le 0.06$
3. **Maximum Target Cap:** $\text{rank} \le 8$
4. **Ordering:** `source1_entity_id ASC`, within-entity `model_score DESC, candidate_entity_id ASC`
5. **Singleton Representation:** Empty string `""` (`S1-xxxxx\t\n`)

---

## 4. Validated Empirical Performance (Fold 0, Full Negative Distribution)

- **Baseline Full-Universe Macro $F_{0.5}$:** `0.761170` (Precision: 0.8266, Recall: 0.6962)
- **Phase 6 Calibrated Policy Macro $F_{0.5}$:** `0.805917` (Precision: 0.9595, Recall: 0.6493)
- **Net Delta in Macro $F_{0.5}$:** **+0.0447 (+4.47 percentage points)**
- **Net Delta in Micro Precision:** **+0.1329 (+13.29 percentage points)**
- **Singleton Accuracy:** **0.9812** (+5.78% over baseline)

---

## 5. Output Submission Artifacts

| Property | Value |
| :--- | :--- |
| **Final Submission Path** | `output/matching_results.tsv` |
| **Mirrored Phase 6 Path** | `P2/predictions/phase6/matching_results.tsv` |
| **Candidate Pairs Path** | `output/candidate_pairs.tsv` |
| **Format** | TSV (`\t` delimiter, UTF-8, no quotes) |
| **Header** | `source1_entity_id\tmatched_entity_ids` |
| **Data Rows** | 1,732,544 |
| **Total Lines** | 1,732,545 |
| **File Size** | 81,238,977 bytes |
| **SHA256 Checksum** | `fce26bfc78d6c2a74829ff512981d94d3b66d89ca5e1d232b14ea1eafa402f9a` |
| **Matched S1 Entities** | 1,539,711 (88.87%) |
| **No-Match S1 Entities (Singletons)** | 192,833 (11.13%) |
| **Total Matched Pairs** | 4,555,480 |
| **S2 Matches** | 2,186,542 |
| **S3 Matches** | 2,368,938 |
| **Validation Result** | **PASS (All structural & domain checks verified)** |

---

## 6. Reproduction Commands

```bash
# Verify baseline integrity
python3 P2/scripts/verify_baseline.py

# Re-run error decomposition and full-universe diagnostics
python3 P2/scripts/phase6_error_decomposition.py

# Re-generate calibrated Phase 6 submission
python3 P2/scripts/phase6_generate_submission.py

# Verify SHA256 checksum
shasum -a 256 output/matching_results.tsv
# Output: fce26bfc78d6c2a74829ff512981d94d3b66d89ca5e1d232b14ea1eafa402f9a
```
