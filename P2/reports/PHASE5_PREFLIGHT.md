# Phase 5 Preflight Report

**Project**: Amazon ML Challenge 2026 — Entity Resolution  
**Document Type**: Preflight Planning Document (DO NOT EXECUTE PHASE 5 YET)  
**Date**: September 27, 2026  
**Status**: Ready for Phase 5 Execution  

---

## 1. Git State

- **Current Branch**: `main`
- **HEAD Commit**: `f21c81d` ("P2 phase4: entity-resolution matching pipeline, models, reports, and test predictions")
- **Preceding Commit**: `152086b` ("commit: phase 3 b")
- **Remote Tracking**: `origin/main` (ahead by 2 commits locally; background LFS upload in progress)
- **Working Tree**: Completely clean (`git status -uall` reports 0 untracked files, 0 unstaged modifications)
- **Phase 3 Files**: Unchanged and frozen

---

## 2. Submission Format

The official submission file format was verified from `output/t090/matching_results.tsv` and `P1/README.md`:

- **File Name**: `matching_results.tsv` (or submission TSV format)
- **Separator**: Tab (`\t`)
- **Header**:
  ```tsv
  source1_entity_id	matched_entity_ids
  ```
- **Row Coverage**: Exactly **1,732,544 rows** (plus 1 header row = 1,732,545 lines). Every test S1 entity from `test_s1_entities.parquet` must appear as a unique row.
- **Representation of Matches**:
  - Comma-delimited list of matched candidate IDs: `"S2-123456,S3-789012"`
  - S2 and S3 predictions are combined into the same comma-delimited string.
  - Sorting within the string does not impact the evaluation metric (the scorer parses to a Python `set`).
- **Representation of No-Match**:
  - Empty string: `""`
  - Explicitly evaluated by the official scorer (`set()` representation).

---

## 3. Official Scorer

- **File Path**: [validation/scorer_v1.py](file:///Users/krishnagera/Amazon-ML-Challenge-2026/validation/scorer_v1.py) (calling [validation/metrics.py](file:///Users/krishnagera/Amazon-ML-Challenge-2026/validation/metrics.py))
- **Interface**:
  ```python
  def score_predictions(
      predictions: Dict[str, Set[str]],
      ground_truth: Dict[str, Set[str]],
      entity_ids: Iterable[str] | None = None,
  ) -> Dict[str, Any]
  ```
- **Evaluation Mechanism**:
  1. The scoring unit is exactly **one Source-1 entity**.
  2. For each S1 entity, precision and recall are computed on `set(predicted)` vs `set(actual)`.
  3. Per-entity $F_{0.5}$ is calculated with $\beta = 0.5$ (weighting precision 2x over recall):
     $$F_{0.5} = \frac{1.25 \cdot \text{precision} \cdot \text{recall}}{0.25 \cdot \text{precision} + \text{recall}}$$
  4. **Special No-Match Cases**:
     - `actual == empty` and `predicted == empty`: Correct no-match $\rightarrow$ Precision = 1.0, Recall = 1.0, $F_{0.5} = 1.0$.
     - `actual == empty` and `predicted != empty`: False match $\rightarrow$ Precision = 0.0, Recall = 0.0, $F_{0.5} = 0.0$.
     - `actual != empty` and `predicted == empty`: Missed match $\rightarrow$ Precision = 0.0, Recall = 0.0, $F_{0.5} = 0.0$.
  5. The competition metric `macro_f0.5` is the unweighted arithmetic mean of $F_{0.5}$ across all evaluated S1 entities.

---

## 4. Existing Submission Code

- `output/t090/matching_results.tsv`: Historical benchmark submission (E06, $T=0.90$) containing 1,732,544 S1 entities.
- `P2/scripts/finish_pipeline.py`: Legacy pipeline script documenting submission aggregation and post-processing architecture.
- `P2/scripts/phase3_to_8_pipeline.py`: Legacy orchestration script documenting post-processing steps.
- `P2/scripts/generate_e03_predictions.py`: Historical candidate-to-submission conversion logic.

---

## 5. Phase 4 Prediction Artifacts

The primary Phase 4 outputs available for Phase 5 consumption are:

1. **`P2/predictions/phase4/test_predictions_s2.tsv`**:
   - File Size: 1,510,836,431 bytes (~1.41 GB)
   - Lines: 43,841,929 (43,841,928 predictions + 1 header)
   - SHA256: `1a4e94cc9b15ba984588db2b763e6c15967b2d7a7ef04213833d79dfe11307af`
   - Schema: `source1_entity_id`, `candidate_entity_id`, `model_score`, `predicted_match_label`
   - Predicted Positives: 5,132,268 (11.7063%)
2. **`P2/predictions/phase4/test_predictions_s3.tsv`**:
   - File Size: 1,774,881,395 bytes (~1.65 GB)
   - Lines: 51,354,668 (51,354,667 predictions + 1 header)
   - SHA256: `503282181cff0df65b86f3f19c2f2813b4cd84d99a1a2b4ed167ac2199d5e3aa`
   - Schema: `source1_entity_id`, `candidate_entity_id`, `model_score`, `predicted_match_label`
   - Predicted Positives: 5,699,683 (11.0987%)
3. **Combined Candidate Coverage**: Exactly 95,196,595 candidate pairs scored (100.0% coverage of Phase 3 V3 test candidates, 0 nulls, 0 duplicate pairs).

---

## 6. Training / OOF Artifacts Available

- `P2/models/phase4/lgb_fold{0..4}.txt`: 5 trained LightGBM booster model binaries (~1.8 MB each).
- `P2/reports/phase4_oof_sweep_results.json`: Full 9-point threshold sweep metrics (0.10 to 0.90) and Fold 0 controlled experiment results.
- `P2/reports/phase4_modeling_metrics.json`: Complete serialized Phase 4 execution metrics, SHA256 hashes, and integrity statistics.
- `P3/reports/folds_v1_manifest.tsv`: Canonical 5-fold S1 split manifest (2,206,821 S1 rows).
- `outputs/person1_step1/train_ground_truth_reconstructed.tsv`: Authoritative ground truth (2,206,821 rows, 7,638,365 true pairs).
- *Raw Pair-Level Training/OOF Prediction Tables*: `NOT FOUND` on disk (purged from memory during training to preserve disk space; aggregated metrics preserved).

---

## 7. Phase 5 Inputs

Phase 5 requires the following exact files:
1. `P2/predictions/phase4/test_predictions_s2.tsv`
2. `P2/predictions/phase4/test_predictions_s3.tsv`
3. `P1/data/entities/test/source1/test_s1_entities.parquet` (to ensure 100% of all 1,732,544 test S1 entities are represented in final output, including zero-match entities)
4. `validation/scorer_v1.py` (for any validation post-processing audits)

---

## 8. Phase 5 Unknowns & Decisions Needed

1. **Post-Processing Aggregation**:
   - Should Phase 5 simply take `predicted_match_label == 1` (equivalent to $T \ge 0.60$), or should top-$K$ per-S1 ranking or calibrated dynamic thresholding be applied?
   - In Phase 4, the unconstrained $T=0.60$ threshold yielded optimal out-of-fold Macro $F_{0.5} = 0.841781$.
2. **Output Location**:
   - Should the final submission be saved to `output/matching_results.tsv` or `P3/submissions/matching_results.tsv`? (Standard is `output/matching_results.tsv`).
3. **Graph Transitive Closure**:
   - Whether any multi-source transitive clustering (e.g. S1-S2 match + S2-S3 match $\rightarrow$ S1-S3 match) should be enforced, or if pairwise predictions are preserved independently.

---

## 9. Recommended Execution Order for Phase 5

1. **Step 1: DuckDB Aggregation Pipeline**:
   - Ingest `test_predictions_s2.tsv` and `test_predictions_s3.tsv` where `predicted_match_label = 1`.
   - Perform `UNION ALL` of positive pairs across S2 and S3.
   - Aggregate by `source1_entity_id`: `string_agg(candidate_entity_id, ',' ORDER BY model_score DESC)`.
2. **Step 2: Full S1 Left-Join**:
   - Left-join all 1,732,544 test S1 entities from `test_s1_entities.parquet` onto the aggregated positive matches.
   - Fill non-matched S1 entities with empty string `""`.
3. **Step 3: Submission TSV Export**:
   - Export strictly formatted TSV `source1_entity_id\tmatched_entity_ids` to `output/matching_results.tsv`.
4. **Step 4: Submission Integrity Verification**:
   - Verify line count is exactly 1,732,545 (1 header + 1,732,544 rows).
   - Verify 0 duplicate S1 rows, 0 nulls, and clean comma formatting.
5. **Step 5: Documentation**:
   - Generate `P3/reports/PHASE5_SUBMISSION_REPORT.md` and compute final SHA256 checksums.

---

## 10. Resource Constraints

- **Total Disk Space**: 460 GiB
- **Used Space**: 400 GiB
- **Available Space**: 30 GiB
- **Predicted Submission Size**: ~50–100 MB (1.73M rows of comma-delimited strings).
- **Execution Overhead**: Aggregation via DuckDB streams in < 60 seconds using < 2 GB RAM.
