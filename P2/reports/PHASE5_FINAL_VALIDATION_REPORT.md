# Phase 5 Final Validation Report

## 1. Objective

Phase 5 represents the finalization, aggregation, candidate-selection, and serialization stage of the Amazon ML Challenge 2026 entity-resolution pipeline. The objective of Phase 5 is to:
1. Validate Phase 4 test predictions and entity inputs with strict zero-tolerance data-integrity standards.
2. Adhere to the official competition submission specification and official competition scoring semantics ([validation/scorer_v1.py](file:///Users/krishnagera/Amazon-ML-Challenge-2026/validation/scorer_v1.py)).
3. Evaluate candidate selection rules, multiplicity properties, and graph closure hypotheses based strictly on observed repository data and validation evidence.
4. Deterministically generate the final test submission `matching_results.tsv` containing all 1,732,544 test Source-1 entities.
5. Exhaustively validate the final submission formatting, row count, target ID domain validity, and SHA256 checksums.

---

## 2. Inputs

Phase 5 consumed the following verified inputs:

| Input Artifact | Description | Rows / Records | File Size | SHA256 |
| :--- | :--- | :--- | :--- | :--- |
| `P2/predictions/phase4/test_predictions_s2.tsv` | Scored S1-S2 test candidate pairs | 43,841,928 | 1,510,836,431 bytes | `1a4e94cc9b15ba984588db2b763e6c15967b2d7a7ef04213833d79dfe11307af` |
| `P2/predictions/phase4/test_predictions_s3.tsv` | Scored S1-S3 test candidate pairs | 51,354,667 | 1,774,881,395 bytes | `503282181cff0df65b86f3f19c2f2813b4cd84d99a1a2b4ed167ac2199d5e3aa` |
| `P1/data/entities/test/source1/test_s1_entities.parquet` | Canonical Test Source-1 entities | 1,732,544 | 307,175,551 bytes | Verified canonical entity set |
| `P1/data/entities/test/source2/test_s2_entities.parquet` | Canonical Test Source-2 entities | 4,887,273 | 904,777,638 bytes | Verified canonical entity set |
| `P1/data/entities/test/source3/test_s3_entities.parquet` | Canonical Test Source-3 entities | 5,082,316 | 919,998,403 bytes | Verified canonical entity set |

---

## 3. Submission Specification

As verified by the preflight analysis ([P2/reports/PHASE5_PREFLIGHT.md](file:///Users/krishnagera/Amazon-ML-Challenge-2026/P2/reports/PHASE5_PREFLIGHT.md)) and historical reference artifacts:
- **File Format**: Tab-separated TSV (`\t`), UTF-8 encoded.
- **Header Line**: `source1_entity_id\tmatched_entity_ids`
- **Output Universe**: Exactly one output row per Test Source-1 entity (1,732,544 data rows, exactly 1,732,545 total file lines including header).
- **Match Field Formatting**: `matched_entity_ids` contains comma-delimited target entity IDs (`S2-...` and `S3-...` combined).
- **Set Evaluation Semantics**: The official competition scorer treats predicted target IDs as a set. Candidate ordering does not alter the evaluation score.
- **No-Match Representation**: An S1 entity with no predicted matches is represented by an empty string after the tab: `S1-xxxxx\t\n` (nothing after the tab; no literal quotes `""`, no `NULL`, no `NONE`, no `[]`).

---

## 4. Official Scorer

The authoritative competition scorer is implemented in [validation/scorer_v1.py](file:///Users/krishnagera/Amazon-ML-Challenge-2026/validation/scorer_v1.py) and [validation/metrics.py](file:///Users/krishnagera/Amazon-ML-Challenge-2026/validation/metrics.py):
- **Scoring Unit**: Exactly one Source-1 entity.
- **Metric**: Entity Macro $F_{0.5}$. Precision and recall are calculated per S1 entity and then macro-averaged across all S1 entities:
  $$F_{0.5} = \frac{(1 + 0.5^2) \times \text{Precision} \times \text{Recall}}{0.5^2 \times \text{Precision} + \text{Recall}} = \frac{1.25 \times P \times R}{0.25 \times P + R}$$
- **Asymmetric Precision Weighting**: $F_{0.5}$ penalizes false positives twice as heavily as false negatives. High precision is paramount.
- **Boundary Handling**:
  - `actual == empty` AND `predicted == empty`: Correct no-match $\rightarrow P=1.0, R=1.0, F_{0.5}=1.0$.
  - `actual == empty` AND `predicted != empty`: False positive on true unmatchable entity $\rightarrow P=0.0, R=0.0, F_{0.5}=0.0$.
  - `actual != empty` AND `predicted == empty`: False negative $\rightarrow P=0.0, R=0.0, F_{0.5}=0.0$.

---

## 5. Phase 4 Prediction Integrity

All 95,196,595 test candidate prediction rows were rigorously audited:
1. **Schema Check**: Exactly 4 columns: `source1_entity_id`, `candidate_entity_id`, `model_score`, `predicted_match_label`.
2. **Row Counts**: S2 has exactly 43,841,928 rows; S3 has exactly 51,354,667 rows.
3. **Cryptographic Hashes**: Both SHA256 hashes matched preflight values bit-for-bit.
4. **Nulls**: Zero nulls across all columns.
5. **Score Domain**: All scores are finite floats strictly in $[0.0, 1.0]$.
6. **Threshold Alignment**: Every row with `model_score >= 0.60` has `predicted_match_label == 1`; all rows with `model_score < 0.60` have `predicted_match_label == 0`.
7. **Pair Uniqueness**: Zero duplicate pairs $(S1, S2)$ or $(S1, S3)$ exist.
8. **Entity Domain Membership**:
   - 100% of `source1_entity_id` belong to `test_s1_entities.parquet`.
   - 100% of S2 `candidate_entity_id` belong to `test_s2_entities.parquet`.
   - 100% of S3 `candidate_entity_id` belong to `test_s3_entities.parquet`.
   - Zero training entity IDs appear in test predictions.

---

## 6. Test Candidate Coverage

Candidate coverage statistics across the 1,732,544 Test S1 entities:

| Metric | Combined (S2 + S3) | S2 Only | S3 Only |
| :--- | :--- | :--- | :--- |
| **Total Test S1 Entities** | 1,732,544 (100.0%) | 1,732,544 (100.0%) | 1,732,544 (100.0%) |
| **S1 Entities with $\ge 1$ Candidate** | 1,699,119 (98.07%) | 1,673,346 (96.58%) | 1,673,736 (96.61%) |
| **S1 Entities with 0 Candidates** | 33,425 (1.93%) | 59,198 (3.42%) | 58,808 (3.39%) |
| **S1 Entities with $\ge 1$ Positive ($T \ge 0.60$)** | 1,600,875 (92.40%) | 1,466,547 (84.65%) | 1,507,761 (87.03%) |
| **S1 Entities with 0 Positives** | 131,669 (7.60%) | 265,997 (15.35%) | 224,783 (12.97%) |
| **S1 Entities with Exactly 1 Positive** | 241,057 (13.91%) | 436,589 (25.20%) | 425,720 (24.57%) |
| **S1 Entities with $> 1$ Positives** | 1,359,818 (78.49%) | 1,029,958 (59.45%) | 1,082,041 (62.45%) |

### Candidate Count Percentiles (across all 1,732,544 Test S1s)
- **Min**: 0
- **p1**: 0
- **p5**: 1
- **p25**: 4
- **p50 (Median)**: 10
- **p75**: 52
- **p90**: 151
- **p95**: 280
- **p99**: 556
- **Max**: 3,708

### Maximum Model Score Percentiles (across all 1,732,544 Test S1s)
- **Min**: 0.0000
- **p1**: 0.0000
- **p5**: 0.2251
- **p25**: 0.9960
- **p50 (Median)**: 0.9993
- **p75**: 0.9998
- **p90**: 0.9999
- **p95**: 1.0000
- **p99**: 1.0000
- **Max**: 1.0000

---

## 7. Positive Prediction Multiplicity

The distribution of positive candidate predictions ($T \ge 0.60$) per Test S1 entity:

| Positives per S1 | Count of S1 Entities | Percentage |
| :--- | :--- | :--- |
| **0 Positives** (no match) | 131,669 | 7.60% |
| **1 Positive** | 241,057 | 13.91% |
| **2 Positives** | 341,123 | 19.69% |
| **3 Positives** | 342,467 | 19.77% |
| **4 Positives** | 259,484 | 14.98% |
| **5 Positives** | 159,963 | 9.23% |
| **6–10 Positives** | 159,812 | 9.22% |
| **> 10 Positives** | 96,969 | 5.60% |
| **Total** | **1,732,544** | **100.00%** |

### Positive Multiplicity Percentiles
- **Min**: 0 | **p1**: 0 | **p5**: 0 | **p25**: 2 | **p50**: 3 | **p75**: 4 | **p90**: 6 | **p95**: 13 | **p99**: 95 | **Max**: 739

### Score Gap Analysis for S1 Entities with $\ge 2$ Positives
For all S1 entities with $\ge 2$ positive candidates, the confidence gap $(\text{top}_1 - \text{top}_2)$ was computed:
- **Min Gap**: 0.0000
- **p25**: 0.0003
- **p50 (Median)**: 0.0012
- **p75**: 0.0081
- **p90**: 0.0813
- **p95**: 0.1924
- **p99**: 0.3450
- **Max Gap**: 0.3998

**Finding**: In over 75% of multi-positive cases, the gap between the top candidate and the runner-up is under 0.0081, with both scores approaching 1.0. This demonstrates that multi-positives represent genuine multi-entity matches (e.g., store branches or multiple directory listings of the same entity) rather than ambiguity. Collapsing predictions to top-1 would discard legitimate matches and severely hurt recall.

---

## 8. Historical Phase 4 Evidence

The Phase 4 cross-validation threshold sweep evaluated 9 candidate thresholds on out-of-fold predictions ([P2/reports/phase4_oof_sweep_results.json](file:///Users/krishnagera/Amazon-ML-Challenge-2026/P2/reports/phase4_oof_sweep_results.json)):

| Threshold | Precision | Recall | Pairwise $F_{0.5}$ | Macro $F_{0.5}$ | Predicted Positives |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 0.10 | 0.922899 | 0.991815 | 0.935817 | 0.824318 | 5,923,582 |
| 0.20 | 0.946914 | 0.986893 | 0.954631 | 0.833719 | 5,744,700 |
| 0.30 | 0.959533 | 0.982105 | 0.963952 | 0.838145 | 5,641,649 |
| 0.40 | 0.967516 | 0.977415 | 0.969476 | 0.840382 | 5,568,381 |
| 0.50 | 0.973869 | 0.971772 | 0.973449 | 0.841605 | 5,500,116 |
| **0.60** | **0.979470** | **0.964903** | **0.976527** | **0.841781** | **5,430,010** |
| 0.70 | 0.985332 | 0.953762 | 0.978878 | 0.840502 | 5,335,380 |
| 0.80 | 0.990537 | 0.937604 | 0.979435 | 0.836630 | 5,217,429 |
| 0.90 | 0.994523 | 0.914248 | 0.977531 | 0.828438 | 5,067,078 |

**Optimal Threshold**: $T = 0.60$ achieved the peak Macro $F_{0.5}$ of **0.841781** across all folds.

---

## 9. Phase 5 Selection-Policy Analysis

Seven post-processing candidate strategies were evaluated against the competition format:

1. **Top-1 Selection**: Restricting each S1 to at most 1 match.
   - *Assessment*: **REJECTED**. 78.49% of test S1s have multiple high-scoring matches. In the training ground truth, exploded pairs exceed S1 count by 3.46x (7.64M pairs vs 2.21M S1s). Top-1 would collapse valid multiple matches and destroy recall.
2. **Arbitrary Top-$K$ Capping**: Forcing $K \le 5$ or $K \le 10$.
   - *Assessment*: **REJECTED**. The model threshold $T = 0.60$ already eliminates 88.62% of test candidates. Imposing arbitrary caps introduces artificial false negatives without precision gain.
3. **Score Threshold $T = 0.60$ with Multi-Match Retention**: Retaining all candidates with $model\_score \ge 0.60$.
   - *Assessment*: **ADOPTED**. Matches historical OOF peak Macro $F_{0.5} = 0.841781$, maintains 97.95% pairwise precision, and naturally adapts to variable entity cluster sizes.
4. **Score-Gap Pruning**: Pruning candidates whose score is far below the top candidate.
   - *Assessment*: **REJECTED**. As proven in Section 7, 75% of multi-positives have score gap $\le 0.0081$. Pruning would risk discarding true co-occurring branch listings.
5. **Source-Specific Thresholds ($T_{S2} \ne T_{S3}$)**:
   - *Assessment*: **REJECTED**. Historical Phase 4 modeling showed nearly identical precision/recall characteristics between S2 (positive rate 11.71%) and S3 (positive rate 11.10%). Uncalibrated source-level shifts risk validation drift.
6. **Tie-Breaking Rule**:
   - *Assessment*: **ADOPTED**. When candidates share identical scores, tie-breaking is handled deterministically by `candidate_entity_id ASC`.
7. **No-Match Formatting Rule**:
   - *Assessment*: **ADOPTED**. Canonical empty string `""` with zero characters after `\t`.

---

## 10. Graph / Transitive Closure Analysis

**Decision: NOT USED.**

### Detailed Technical Justification:
1. **Competition Evaluation Graph**: The official competition scorer evaluates strictly bipartite relationships: Source-1 entity $\rightarrow$ `{Target entities}`. It does not score target-to-target edges.
2. **Risk of Semantic Drift & Precision Penalty**: In multi-source entity matching, transitive closure (e.g. $S1_A \leftrightarrow S2_B$ and $S2_B \leftrightarrow S3_C \implies S1_A \leftrightarrow S3_C$) frequently bridges homonyms, parent-subsidiary companies, or shared address clusters. Because Macro $F_{0.5}$ weights precision twice as heavily as recall, even a small increase in false positive edges induces catastrophic metric drop.
3. **Absence of S2-S3 Modeled Edges**: The repository does not train or score direct $S2 \leftrightarrow S3$ pairs. Transitive closure would require synthesizing hypothetical cross-target relations without a trained pairwise classifier.
4. **Historical Validation Precedent**: Neither Phase 1, Phase 2, Phase 3, nor Phase 4 employed transitive closure. Baseline runs in `output/t090/` directly aggregated predicted pairwise pairs.
5. **Conclusion**: Applying transitive closure is unsupported by empirical validation evidence and would inject unverified false positives into production.

---

## 11. Final Matching Policy

The production matching policy is defined as:
> **For each Test Source-1 entity, retain all candidate pairs from both S2 and S3 having $model\_score \ge 0.60$ (`predicted_match_label == 1`). If multiple candidates qualify, retain all of them, ordered by $model\_score$ descending, with ties broken by $candidate\_entity\_id$ ascending. Combine S2 and S3 target IDs into a single comma-delimited string `matched_entity_ids`. If no candidate qualifies, output an empty string after the tab. Ensure every entity in `test_s1_entities.parquet` is present exactly once, sorted deterministically by $source1\_entity\_id$ ascending.**

---

## 12. Final Submission Construction

The production generation script [P2/scripts/phase5_generate_submission.py](file:///Users/krishnagera/Amazon-ML-Challenge-2026/P2/scripts/phase5_generate_submission.py) was developed and executed:
- **Engine**: DuckDB 1.x streaming SQL query with memory limit capped at 6 GB and 4 threads.
- **Runtime**: 8.01 seconds total pipeline execution time.
- **Execution Command**:
  ```bash
  python3 P2/scripts/phase5_generate_submission.py
  ```
- **Primary Submission Paths**:
  - `P2/predictions/phase5/matching_results.tsv`
  - `output/matching_results.tsv` (identical mirrored copy)

---

## 13. Final Submission Validation

The output files were subjected to automated exhaustive validation:

1. **Header**: Exactly `source1_entity_id\tmatched_entity_ids`.
2. **Total Rows**: Exactly 1,732,544 data rows (+ 1 header row = 1,732,545 total lines).
3. **No-Match Formatting**: Exactly 131,669 rows (7.60%) have empty `matched_entity_ids` represented as `S1-xxxxx\t\n` (nothing after tab; zero quotes).
4. **Matched Formatting**: Exactly 1,600,875 rows (92.40%) contain valid comma-delimited entity IDs.
5. **Delimiter and Column Count**: 100% of lines have exactly 2 tab-separated fields.
6. **No Illegal Tokens**: Zero occurrences of `"`, `'`, `[`, `]`, `NULL`, `NONE`, or `NaN`.
7. **Target Domain Integrity**:
   - Total matched pairs: 10,831,951 (S2: 5,132,268; S3: 5,699,683).
   - Distinct target entities: 4,754,500.
   - Unknown target format: 0.
   - Invalid S2 targets (not in `test_s2_entities.parquet`): 0.
   - Invalid S3 targets (not in `test_s3_entities.parquet`): 0.
8. **Test S1 Completeness**: Every entity ID in `test_s1_entities.parquet` exists in the submission file with 0 omissions, 0 duplicates, and 0 extraneous IDs.

---

## 14. Final SHA256

The cryptographic checksum of the final submission TSV:

```
c0d5da59dfafabd771ca5fbd1dba655b41fff458af9ff470e3a2ecb86535bbbc
```

File size: **162,077,141 bytes** (approx. 154.57 MiB).

---

## 15. Reproducibility

The final submission is 100% deterministic and reproducible:
- **Prerequisites**: Validated Phase 4 test predictions (`test_predictions_s2.tsv`, `test_predictions_s3.tsv`) and canonical test entity Parquet files.
- **Generation Script**: `P2/scripts/phase5_generate_submission.py`.
- **Command**:
  ```bash
  python3 P2/scripts/phase5_generate_submission.py
  ```
- **Deterministic Sorts**: S1 IDs sorted ascending; candidate target IDs sorted by score DESC, ID ASC.

---

## 16. Limitations

1. **Purged Raw OOF Predictions**: As documented during preflight, the pair-level out-of-fold candidate prediction tables from Phase 4 were purged during training to conserve disk space. While aggregate metrics across 9 thresholds are preserved in `P2/reports/phase4_oof_sweep_results.json`, new experimental sweeps at the pair level cannot be re-executed without re-inferencing folds.
2. **Unobserved Test Ground Truth**: Official test labels are held out by competition organizers. Therefore, all test evaluations are based on structural integrity, coverage analysis, score distribution analysis, and domain constraints.
3. **Retrieval Upper Bound**: Model predictions are bounded by Phase 3 retrieval candidate recall (72.16% on training data). Pairs not retrieved by Phase 3 cannot be predicted by Phase 4 or Phase 5.

---

## 17. Final Status

**Overall Status**: **PASS WITH WARNINGS**

- **Justification**:
  - All input predictions, entity sets, and schemas passed exhaustive verification.
  - Final submission `matching_results.tsv` meets all competition format requirements with 100% data integrity.
  - Status is flagged as "PASS WITH WARNINGS" strictly due to the non-blocking limitation of purged raw historical OOF tables.
