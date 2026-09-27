# Phase 5 Handoff

**FINAL SUBMISSION:**
`/Users/krishnagera/Amazon-ML-Challenge-2026/output/matching_results.tsv`  
(also mirrored at `/Users/krishnagera/Amazon-ML-Challenge-2026/P2/predictions/phase5/matching_results.tsv`)

**FORMAT:**
TSV (UTF-8, tab-separated, header `source1_entity_id\tmatched_entity_ids`, no quotes, empty string for no-match)

**ROWS:**
`1,732,544` data rows (exactly `1,732,545` total lines including header)

**SHA256:**
`c0d5da59dfafabd771ca5fbd1dba655b41fff458af9ff470e3a2ecb86535bbbc`

**THRESHOLD:**
`0.60`

**FINAL MATCHING POLICY:**
For each Test Source-1 entity, retain all candidate pairs from both S2 and S3 having `model_score >= 0.60` (`predicted_match_label == 1`). If multiple candidates qualify, retain all of them, ordered by `model_score` descending, with ties broken by `candidate_entity_id` ascending. Combine S2 and S3 target IDs into a single comma-delimited string `matched_entity_ids`. If no candidate qualifies, output an empty string after the tab (`S1-xxxxx\t\n`). Ensure every entity in `test_s1_entities.parquet` is present exactly once, sorted deterministically by `source1_entity_id` ascending.

**SCORER:**
`validation/scorer_v1.py` (Entity-level Macro $F_{0.5}$)

**INPUT PREDICTIONS:**

- **S2:**
  - Path: `/Users/krishnagera/Amazon-ML-Challenge-2026/P2/predictions/phase4/test_predictions_s2.tsv`
  - Rows: `43,841,928`
  - SHA256: `1a4e94cc9b15ba984588db2b763e6c15967b2d7a7ef04213833d79dfe11307af`

- **S3:**
  - Path: `/Users/krishnagera/Amazon-ML-Challenge-2026/P2/predictions/phase4/test_predictions_s3.tsv`
  - Rows: `51,354,667`
  - SHA256: `503282181cff0df65b86f3f19c2f2813b4cd84d99a1a2b4ed167ac2199d5e3aa`

**VALIDATION STATUS:**
`PASS WITH WARNINGS`

**REPRODUCTION COMMAND:**
```bash
python3 P2/scripts/phase5_generate_submission.py
```

**LIMITATIONS:**
1. Raw pair-level out-of-fold candidate prediction tables from Phase 4 training were purged to conserve disk space, preventing new fine-grained pair post-processing sweeps on historical validation folds. The 9-point threshold sweep summary is fully preserved in `P2/reports/phase4_oof_sweep_results.json` (Macro $F_{0.5} = 0.841781$).
2. Test split ground truth is held out by competition organizers; validation is based on strict schema conformity, domain membership checks, and distribution analyses.
3. Candidate coverage is capped by Phase 3 retrieval recall (72.16% on training data).
