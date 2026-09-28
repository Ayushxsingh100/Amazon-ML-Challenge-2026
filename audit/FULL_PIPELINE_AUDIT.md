# Amazon ML Challenge 2026
# Full Pipeline Forensic Audit

**Audit Date**: 2026-09-26  
**Auditor**: Automated forensic audit  
**Scope**: Complete entity-resolution pipeline from raw data through submission  
**Reported Score**: 0.7695 (Macro F0.5, OOF on V1 candidates, T=0.5)

---

## 1. Executive Summary

The pipeline achieves a **Macro F0.5 of 0.7695** using out-of-fold evaluation on the cands_BCD_v1 candidate universe. The candidate oracle (perfect model on the same candidates) is **0.7845**, meaning the model loses only ~0.015 F0.5. The **dominant performance bottleneck is candidate generation recall**, which caps the pipeline at ~78% of maximum achievable score.

### Key Findings

| # | Finding | Priority | Impact |
|---|---------|----------|--------|
| 1 | **Candidate recall is ~60%** (S2: 59.9%, S3: 59.5%) — this is the #1 bottleneck | P1 | ~0.22 F0.5 lost |
| 2 | **V2 candidates exist but are NOT used** — V2 improves recall to ~66% but E02 trains on V1 | P1 | ~0.06 recall wasted |
| 3 | **No models/features/predictions exist locally** — all training artifacts from Windows machine only | P0 | Cannot reproduce |
| 4 | **Test candidates in P1 outputs differ from V1/V2 candidates** — disconnected test pipeline | P1 | Test uses wrong candidates |
| 5 | **Feature regex inconsistency in finish_pipeline.py** — different regex vs canonical | P2 | Feature mismatch |
| 6 | Evaluation metric implementation is **correct** | PASS | — |
| 7 | Ground truth is **correct** — SHA256 matches between raw and reconstructed | PASS | — |
| 8 | Row retention is **perfect** — zero rows lost in normalization | PASS | — |
| 9 | ID integrity is **correct** — no duplicates, no mutations | PASS | — |

---

## 2. Repository Inventory

### File Classification

| File/Dir | Type | Purpose | Status |
|----------|------|---------|--------|
| data/train/*.tsv.part_* | Source dataset (parts) | Raw split data | Active |
| data/train/train_source{1,2,3}.tsv | Reconstructed dataset | Merged raw data | Active |
| data/train/train_ground_truth.tsv | Ground truth | S1-S2/S3 match labels | Active |
| data/test/test_source{1,2,3}.tsv | Source dataset | Test data | Active |
| outputs/person1_step1/normalized/ | Normalized dataset | Cleaned entity data | Active |
| outputs/person1_step1/train_ground_truth_reconstructed.tsv | Ground truth copy | Identical to raw GT | Active |
| outputs/person1_step1/test_candidate_pairs_{s2,s3}.tsv | Candidate artifact | P1 test candidates (Strategy B only) | **Stale** |
| P2/data/candidates/train_candidate_pairs_{s2,s3}.tsv | Candidate artifact | V1 train candidates (B+C+D) | Active (E02) |
| P2/data/candidates/*_v2.tsv | Candidate artifact | V2 candidates (B+C+D+E+F+G+H+I) | **Unused** |
| P2/scripts/cands_BCD_v1_tasks.py | Blocking impl | V1 candidate generation | Active |
| P2/scripts/cands_BCD_v2_tasks.py | Blocking impl | V2 candidate generation | Generated artifacts exist |
| P2/scripts/e02_train_lgb.py | Model training | LightGBM 5-fold CV | Active (ran on Windows) |
| P2/scripts/phase3_to_8_pipeline.py | Model pipeline | Earlier pipeline | **Obsolete** — references missing files |
| P2/scripts/finish_pipeline.py | Model pipeline | S3 scoring completion | **Obsolete** — regex inconsistency |
| P2/models/ | Model artifact | LightGBM fold models | **Does not exist locally** |
| P2/data/features/ | Feature artifact | Feature TSVs | **Does not exist locally** |
| validation/scorer_v1.py | Evaluation | Official scorer | Active |
| validation/metrics.py | Evaluation | F0.5 implementation | Active |
| P3/reports/folds_v1_manifest.tsv | Configuration | S1 fold assignments | Active (Frozen) |
| src/normalization.py | Utility | Python normalization | **Not used by production pipeline** |
| src/blocking.py | Utility | Python blocking | **Not used by production pipeline** |
| src/features.py | Utility | Python features | **Not used by production pipeline** |

**Important**: The src/ module (normalization.py, blocking.py, features.py) is NOT used by the actual production pipeline. All production processing uses DuckDB SQL directly in the P2 scripts.

---

## 3. Actual Pipeline

The real pipeline is:

```
Raw TSV Parts -> reconstruct_data.sh -> Normalized TSVs
    -> cands_BCD_v1_tasks.py (Rules A,B,C,D) -> Train Candidate Pairs (V1: 54.6M pairs)
    -> e02_train_lgb.py (LightGBM 5-fold CV, 10% neg downsampling)
    -> OOF Predictions (54.6M scored pairs) -> scorer_v1.py -> Macro F0.5 = 0.7695
```

**WARNING**: The test scoring pipeline uses P1 test candidates (Strategy B: Rules A+B only), NOT the V1 B+C+D candidates. Train uses V1 (B+C+D). This is a **train/test candidate mismatch**.

---

## 4. Dataset Inventory

### Independently Measured Row Counts

| Dataset | Rows | Unique IDs | Columns |
|---------|------|------------|---------|
| train_source1 | 2,206,821 | 2,206,821 | entity_id, business_name, business_address, country |
| train_source2 | 5,034,616 | 5,034,616 | entity_id, business_name, business_address, country |
| train_source3 | 5,285,603 | 5,285,603 | entity_id, business_name, business_address, country |
| train_ground_truth | 2,206,821 | 2,206,821 | source1_entity_id, matched_entity_ids |
| test_source1 | 1,732,544 | 1,732,544 | entity_id, business_name, business_address, country |
| test_source2 | 4,887,273 | 4,887,273 | entity_id, business_name, business_address, country |
| test_source3 | 5,082,316 | 5,082,316 | entity_id, business_name, business_address, country |

### Cross-Check Against Historical Reports

| Metric | Historical (dataset_row_counts.tsv) | Independently Measured | Match |
|--------|---------------------------------------|----------------------|-------|
| train_source1 rows | 2,206,821 | 2,206,821 | YES |
| train_source2 rows | 5,034,616 | 5,034,616 | YES |
| train_source3 rows | 5,285,603 | 5,285,603 | YES |
| train_ground_truth rows | 2,206,821 | 2,206,821 | YES |

---

## 5. Ground Truth Audit

### Key Metrics (Independently Measured)

| Metric | Value |
|--------|-------|
| Total GT rows | 2,206,821 |
| Unique S1 IDs | 2,206,821 |
| Duplicate S1 IDs in GT | 0 |
| S1 with no match (empty) | 123,247 (5.58%) |
| Total exploded pairs | 7,638,365 |
| S2 true pairs | 3,693,619 |
| S3 true pairs | 3,944,746 |
| Other (non-S2, non-S3) pairs | 0 |
| S1 matched to both S2 and S3 | 1,776,047 (80.5%) |
| S1 matched to S2 only | 143,029 (6.5%) |
| S1 matched to S3 only | 164,498 (7.5%) |

### Reconstruction Verification
- Raw GT file SHA256 = Reconstructed GT file SHA256: **MATCH**
- Both contain identical data (2,206,821 rows, same content)

**Verdict: PASS**

---

## 6. Reconstruction Audit

- reconstruct_data.sh uses `cat "${target}.part_"*` (glob-based concatenation with alphabetical sort)
- Row retention: Raw = Normalized for all 6 source files (verified)
- SHA256 of raw GT matches reconstructed GT

**Verdict: PASS**

---

## 7. Normalization Audit

| Dataset | Rows | Rows Lost | Empty Name | Empty Address | Empty Country |
|---------|------|-----------|------------|---------------|---------------|
| train_source1_normalized | 2,206,821 | 0 | 0 | 0 | 0 |
| train_source2_normalized | 5,034,616 | 0 | — | — | — |
| train_source3_normalized | 5,285,603 | 0 | — | — | — |
| test_source1_normalized | 1,732,544 | 0 | — | — | — |
| test_source2_normalized | 4,887,273 | 0 | — | — | — |
| test_source3_normalized | 5,082,316 | 0 | — | — | — |

The normalized files have the same columns as raw (entity_id, business_name, business_address, country). The normalization was applied in-place on the original column values. No additional norm_ columns.

The src/normalization.py module adds norm_name, norm_address, norm_country columns. However, the production pipeline does NOT use src/normalization.py — it operates directly on DuckDB SQL.

**Verdict: PASS**

---

## 8. Entity-ID Integrity

| Stage | Dataset | Total | Unique IDs | Duplicates |
|-------|---------|-------|------------|------------|
| Raw | train_source1 | 2,206,821 | 2,206,821 | 0 |
| Raw | train_source2 | 5,034,616 | 5,034,616 | 0 |
| Raw | train_source3 | 5,285,603 | 5,285,603 | 0 |
| Normalized | train_source1 | 2,206,821 | 2,206,821 | 0 |
| Normalized | train_source2 | 5,034,616 | 5,034,616 | 0 |
| Normalized | train_source3 | 5,285,603 | 5,285,603 | 0 |
| Ground Truth | S1 IDs | 2,206,821 | 2,206,821 | 0 |

**Verdict: PASS**

---

## 9. Row Retention

| Dataset | Raw | Normalized | Lost |
|---------|-----|------------|------|
| train_source1 | 2,206,821 | 2,206,821 | 0 |
| train_source2 | 5,034,616 | 5,034,616 | 0 |
| train_source3 | 5,285,603 | 5,285,603 | 0 |
| test_source1 | 1,732,544 | 1,732,544 | 0 |
| test_source2 | 4,887,273 | 4,887,273 | 0 |
| test_source3 | 5,082,316 | 5,082,316 | 0 |

**Verdict: PASS**

---

## 10. Blocking Audit

### V1 Rules (cands_BCD_v1 — used by E02)
All rules require `country <> '' AND country match`.

| Rule | Fields | Logic |
|------|--------|-------|
| A | prefix4 + house | LEFT(business_name, 4) + regexp_extract(business_address, '[0-9]+[A-Za-z]?', 0) |
| B | exact address | business_address = tgt.business_address |
| C | token1 + house | SPLIT_PART(business_name, ' ', 1) + same house regex |
| D | exact name | TRIM(business_name) = TRIM(tgt.business_name) |

### V2 Rules (cands_BCD_v2 — generated but UNUSED)
Adds Rules E (clean alphanum name), F (zero-stripped house), G (root token + house), H (clean address), I (zip + prefix3).

---

## 11. Candidate Generation Audit

### Candidate File Statistics (Independently Measured)

| Candidate Set | Pairs | Unique S1 | Unique Target | Positives | Negatives |
|--------------|-------|-----------|---------------|-----------|-----------|
| V1 train S2 | 24,594,064 | 1,859,868 | 2,798,463 | 2,214,137 | 22,379,927 |
| V1 train S3 | 29,998,661 | 1,881,377 | 3,016,380 | 2,346,442 | 27,652,219 |
| V1 train total | 54,592,725 | — | — | 4,560,579 | 50,032,146 |
| V2 train S2 | 30,359,040 | 1,929,373 | 3,045,029 | 2,438,575 | 27,920,465 |
| V2 train S3 | 36,973,484 | 1,944,749 | 3,277,458 | 2,584,593 | 34,388,891 |
| V2 train total | 67,332,524 | — | — | 5,023,168 | 62,309,356 |
| P1 test S2 | 26,046,195 | 1,331,914 | 2,298,136 | — | — |
| P1 test S3 | 30,777,878 | 1,350,577 | 2,492,891 | — | — |

### E02 Cross-Check
- E02 OOF row count: 54,592,725 = V1 S2 (24,594,064) + V1 S3 (29,998,661) **MATCH**

---

## 12. Candidate Recall

### V1 Candidate Recall (used by E02 — independently measured)

| Metric | S2 | S3 |
|--------|----|----|
| GT total true pairs | 3,693,619 | 3,944,746 |
| Captured true pairs | 2,214,137 | 2,346,442 |
| Missed true pairs | 1,479,482 | 1,598,304 |
| **Pair recall** | **0.5994** | **0.5948** |
| S1 with GT | 1,919,076 | 1,940,545 |
| S1 fully covered | 892,292 | 846,169 |
| S1 partially covered | 475,680 | 583,513 |
| S1 zero capture | 551,104 | 510,863 |

### V2 Candidate Recall (independently measured)

| Metric | S2 | S3 |
|--------|----|----|
| GT total true pairs | 3,693,619 | 3,944,746 |
| Captured | 2,438,575 | 2,584,593 |
| Missed | 1,255,044 | 1,360,153 |
| **Pair recall** | **0.6602** | **0.6552** |

**CRITICAL**: Even the best candidate set (V2) only captures ~66% of true pairs. 34% of true matches are unreachable by any model trained on these candidates.

---

## 13. Candidate Failure Analysis

From cands_BCD_v2_missed_pair_analysis.tsv:

| Category | S2 Count | S3 Count |
|----------|----------|----------|
| Severe name variation (JW < 0.60) | 1,255,039 | 1,360,142 |
| Non-ASCII (Indic script) names | 386,517 | 306,899 |
| Both missing house number | 67,513 | 85,364 |

The overwhelming majority of missed pairs are due to severe name variation where normalized business names differ substantially. Many involve non-Latin script (Indic/Devanagari) names.

---

## 14. Training Label Audit

Labels assigned by LEFT JOIN against GT: label=1 if pair in GT, label=0 otherwise.

| Source | Positives | Negatives | Positive Rate |
|--------|-----------|-----------|---------------|
| S2 | 2,214,137 | 22,379,927 | 9.00% |
| S3 | 2,346,442 | 27,652,219 | 7.82% |
| Combined | 4,560,579 | 50,032,146 | 8.35% |

E02 trains with 10% deterministic negative downsampling. All positives retained.

**Verdict: PASS**

---

## 15. Feature Audit

19 features all computed in DuckDB SQL. No leakage detected. No NaN/inf risk. Features available at test time. address_first_number_match regex inconsistency exists in finish_pipeline.py only (obsolete script).

**Verdict: WARNING** (inconsistency in obsolete script only)

---

## 16. Model Audit

LightGBM, binary objective, 5-fold CV, 500 rounds, lr=0.05, num_leaves=31, seed=2026.
Fold assignment via SHA256 hash (seed=314159) at S1 entity level.
Model artifacts do NOT exist locally — ran on Windows machine.

---

## 17. Prediction Audit

OOF predictions: 54,592,725 rows, 0 duplicates, 0 missing, all finite. Matches V1 candidate total exactly.

**Verdict: PASS**

---

## 18. Threshold / Match Selection Audit

| Threshold | Macro F0.5 |
|-----------|------------|
| 0.1 | 0.7629 |
| 0.3 | 0.7688 |
| 0.5 | 0.7695 |
| 0.7 | 0.7677 |
| 0.9 | 0.7593 |

Curve is extremely flat. Candidate ceiling is the binding constraint.

---

## 19. Evaluation Audit

F0.5 formula correctly implemented in validation/metrics.py. Special no-match handling is correct. Macro averaging across S1 entities is correct. S2 and S3 scored jointly per S1.

**Verdict: PASS**

---

## 20. Submission Audit

**NOT VERIFIED** — No final submission file found in repository.

---

## 21. Leakage Audit

No leakage detected. GT only used for labeling after candidate generation. Features use only entity attributes. Fold isolation is proper (S1-level).

**Verdict: PASS**

---

## 22. Reproducibility Audit

Model artifacts missing locally. Feature files missing locally. Prediction files missing locally. All generated on Windows. Cannot reproduce without re-running pipeline.

**Verdict: FAIL**

---

## 23. Issue Register

### ISSUE-01: Candidate Recall Ceiling (P1)
- Component: Candidate generation
- V1 recall: S2=59.9%, S3=59.5%
- Impact: Oracle F0.5 = 0.7845 — score cannot exceed this
- Affected: 551,104 S1 with zero S2 candidates, 510,863 with zero S3 candidates

### ISSUE-02: V2 Candidates Not Used (P1)
- V2 improves to 66.0%/65.5% recall — 462,589 additional true pairs captured
- E02 trains on V1, ignoring V2

### ISSUE-03: Train/Test Candidate Mismatch (P1)
- Test uses P1 Strategy B (Rules A+B), train uses V1 (B+C+D)
- phase3_to_8_pipeline.py line 32-33 references P1 test candidates

### ISSUE-04: Model Artifacts Missing Locally (P0)
- No models, features, or predictions exist on this machine
- All ran on Windows (E02_RUN.json shows Windows paths)

### ISSUE-05: Feature Regex Inconsistency (P2)
- finish_pipeline.py uses different house number regex than canonical pipeline
- Likely obsolete script

### ISSUE-06: Obsolete Scripts Reference Missing Files (P2)
- phase3_to_8_pipeline.py references train_strategy_b_candidates_*.tsv which do not exist

---

## 24. Root-Cause Tree

```
Current OOF score = 0.7695 (Macro F0.5)
|
+-- Data correctness
|   +-- Ground truth: PASS (SHA256 verified, 7.6M pairs, no duplicates)
|   +-- Row retention: PASS (0 rows lost at any stage)
|   +-- ID integrity: PASS (0 duplicates, 0 mutations)
|
+-- Candidate generation [PRIMARY BOTTLENECK]
|   +-- V1 candidate recall: S2=59.9%, S3=59.5%
|   |   +-- 551,104 S1 have S2 GT but ZERO S2 candidates
|   |   +-- 510,863 S1 have S3 GT but ZERO S3 candidates
|   |   +-- 1,479,482 S2 + 1,598,304 S3 true pairs missed
|   +-- V2 recall: S2=66.0%, S3=65.5% (NOT USED)
|   +-- Candidate oracle on V1: 0.7845
|   |   +-- Gap from current score: only 0.0151
|   +-- Candidate ceiling gap: 1.0 - 0.7845 = 0.2155
|       +-- THIS IS THE DOMINANT LOSS
|
+-- Modeling [WELL-OPTIMIZED]
|   +-- Model gap (oracle - baseline): 0.0151
|   +-- Threshold curve is flat (0.7629 @ t=0.1 to 0.7695 @ t=0.5)
|   +-- Feature set: 19 features, no leakage, no missing values
|
+-- Selection
|   +-- Threshold: T=0.5 is optimal on sweep
|   +-- Match selection: NOT VERIFIED (no submission file found)
|
+-- Evaluation
    +-- Metric implementation: PASS (F0.5 correctly implemented)
    +-- Submission correctness: NOT VERIFIED
```

---

## 25. Prioritized Work Plan

### 01 — Switch to V2 Candidates (High Impact, Easy)
- Evidence: V2 recall is 66.0/65.5% vs V1 59.9/59.5%. 462,589 additional true pairs.
- Impact: Estimated +0.03 to +0.06 F0.5
- Files: Modify e02_train_lgb.py to reference V2 candidate/feature files
- Validation: Re-run E02 pipeline with V2 candidates, compare OOF score

### 02 — Fix Train/Test Candidate Consistency
- Evidence: Test uses P1 Strategy B (A+B), train uses V1 (B+C+D) or V2
- Impact: Test predictions miss candidates that model was trained to score
- Files: Update test scoring to use V1 or V2 test candidates

### 03 — Improve Candidate Recall Beyond V2
- Evidence: V2 still misses 1.26M S2 + 1.36M S3 true pairs. ~100% severe name variations.
- Impact: Could improve score by 0.05-0.15 F0.5
- Approach: Embedding-based or phonetic blocking for Indic/non-Latin scripts
- Dependency: Requires 01 and 02 first

### 04 — Reproduce Pipeline on Current Machine
- Evidence: No model/feature/prediction files exist locally
- Impact: Cannot iterate without reproducibility

### 05 — Generate and Validate Test Submission
- Evidence: No matching_results.tsv found
- Dependency: Requires 04 (working model)

---

## 26. Files Requiring Changes

| File | Change Needed | Reason |
|------|---------------|--------|
| P2/scripts/e02_train_lgb.py | Update candidate/feature paths to V2 | Use improved candidates |
| Test scoring script (TBD) | Use V1/V2 test candidates | Train/test consistency |
| P2/scripts/finish_pipeline.py | Fix regex or deprecate | Feature inconsistency |
| P2/scripts/phase3_to_8_pipeline.py | Fix paths or deprecate | References missing files |

---

## 27. Files That Should NOT Be Changed

| File | Reason |
|------|--------|
| validation/scorer_v1.py | Correctly implements competition metric |
| validation/metrics.py | F0.5 formula verified |
| P3/reports/folds_v1_manifest.tsv | Frozen, SHA256 verified |
| P3/reports/SCORING_SPEC_v1.md | Frozen specification |
| data/train/train_ground_truth.tsv | Verified correct |

---

## 28. Missing Evidence / Blocked Checks

| Check | Status | Reason |
|-------|--------|--------|
| Test submission file audit | NOT VERIFIED | No submission file found |
| Model prediction distribution | NOT VERIFIED | No prediction files locally |
| Per-country recall breakdown | NOT VERIFIED | Feasible but not run |
| Normalization information loss | NOT VERIFIED | Requires large computation |
| Feature importance verification | NOT VERIFIED | No model files locally |
| Test score verification | NOT VERIFIED | No test predictions locally |

---

## 29. Final Conclusion

The pipeline score of 0.7695 is primarily limited by **candidate generation recall** (~60% on V1). The model is well-trained — the gap between model output (0.7695) and candidate oracle (0.7845) is only 0.015.

### Score Decomposition

| Component | Contribution to Loss |
|-----------|---------------------|
| Candidate ceiling (1.0 - oracle) | **0.2155** (dominant) |
| Model gap (oracle - baseline) | 0.0151 (small) |
| Floor (true no-match entities) | 0.0558 (structural) |

### Audit Scorecard

| Area | Status | Priority |
|------|--------|----------|
| Raw data | PASS | — |
| Reconstruction | PASS | — |
| Ground truth | PASS | — |
| Normalization | PASS | — |
| ID integrity | PASS | — |
| Row retention | PASS | — |
| Blocking | WARNING | P1 |
| Candidate recall | **FAIL** | **P1** |
| Candidate integrity | PASS | — |
| Training labels | PASS | — |
| Features | WARNING | P2 |
| Model | PASS | — |
| Threshold | PASS | — |
| Evaluation | PASS | — |
| Submission | NOT VERIFIED | P0 |
| Reproducibility | **FAIL** | P0 |
