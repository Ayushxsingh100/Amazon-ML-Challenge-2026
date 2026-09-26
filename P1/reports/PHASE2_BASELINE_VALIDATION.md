# Phase 2 — Candidate Baseline Validation Report

**Author**: Person 1 (P1)  
**Date**: September 26, 2026  
**Repository**: [Amazon-ML-Challenge-2026](https://github.com/Ayushxsingh100/Amazon-ML-Challenge-2026.git)  
**Scope**: Independent validation of candidate generation rules, artifact provenance, evaluation framework, and candidate distribution diagnostics for V1, V2, and test baselines.

---

## 1. Executive Summary

Phase 2 established an independent, deterministic candidate set evaluation framework ([evaluate_candidate_set.py](file:///Users/krishnagera/Amazon-ML-Challenge-2026/P1/scripts/evaluation/evaluate_candidate_set.py)) and performed a forensic validation of all 10 candidate pair files present in the repository.

### Key Forensic Findings
1. **Recall Verification**:
   - **V1 Baseline**: Exactly reproduced. S2 Recall = **59.9449%** (2,214,137 / 3,693,619), S3 Recall = **59.4827%** (2,346,442 / 3,944,746), Combined Recall = **59.7062%** (4,560,579 / 7,638,365).
   - **V2 Baseline**: Exactly reproduced. S2 Recall = **66.0213%** (2,438,575 / 3,693,619), S3 Recall = **65.5199%** (2,584,593 / 3,944,746), Combined Recall = **65.7623%** (5,023,168 / 7,638,365).
   - **Phase 0 Parity**: Zero numerical discrepancy with Phase 0 report and verified cache.
2. **Cryptographic Provenance**:
   - Stored V2 candidate files match the file-level SHA256 hashes recorded in [P2/reports/cands_BCD_v2_manifest.tsv](file:///Users/krishnagera/Amazon-ML-Challenge-2026/P2/reports/cands_BCD_v2_manifest.tsv) with 100% bit-for-bit fidelity.
   - Stored V1 candidate files match the sorted pair deterministic SHA256 hashes recorded in [P2/reports/cands_BCD_v1_manifest.tsv](file:///Users/krishnagera/Amazon-ML-Challenge-2026/P2/reports/cands_BCD_v1_manifest.tsv) with 100% fidelity.
3. **Data Quality & Schema Cleanliness**:
   - **0 duplicate pairs** across all 10 candidate files (totaling 319,258,409 candidate rows evaluated).
   - **0 null IDs**, **0 self-matches**, **0 malformed IDs**.
   - **0 invalid entity IDs** (every candidate ID exists in the canonical P1 entity pool).
   - **0 cross-country candidate pairs** (blocking strictly enforces country equivalence).
4. **Train / Test Parity Critical Finding**:
   - [P2/scripts/cands_BCD_v1_tasks.py](file:///Users/krishnagera/Amazon-ML-Challenge-2026/P2/scripts/cands_BCD_v1_tasks.py) and [P2/scripts/cands_BCD_v2_tasks.py](file:///Users/krishnagera/Amazon-ML-Challenge-2026/P2/scripts/cands_BCD_v2_tasks.py) generate test candidate sets using rules identical to train.
   - However, legacy downstream scoring scripts ([finish_pipeline.py](file:///Users/krishnagera/Amazon-ML-Challenge-2026/P2/scripts/finish_pipeline.py) and [phase3_to_8_pipeline.py](file:///Users/krishnagera/Amazon-ML-Challenge-2026/P2/scripts/phase3_to_8_pipeline.py)) consumed `outputs/person1_step1/test_candidate_pairs_s*.tsv` (Strategy B: Rules A+B only). This caused 400,630 test S1 entities (24.21%) to have zero candidates, severely degrading downstream leaderboard predictions.

---

## 2. Actual V1 Implementation

The canonical V1 generator is [P2/scripts/cands_BCD_v1_tasks.py](file:///Users/krishnagera/Amazon-ML-Challenge-2026/P2/scripts/cands_BCD_v1_tasks.py).

### Input Files
- Normalized entity tables: `outputs/person1_step1/normalized/{train,test}_source{1,2,3}_normalized.tsv`
- Ground truth: `outputs/person1_step1/train_ground_truth_reconstructed.tsv`

### Normalization Fields Extracted In-Query
```sql
LEFT(TRIM(business_name), 4) AS prefix4,
split_part(TRIM(business_name), ' ', 1) AS token1,
regexp_extract(TRIM(business_address), '[0-9]+[A-Za-z]?', 0) AS house
```

### Exact Blocking Rules
- **Rule A**: `s1.country = tgt.country AND s1.prefix4 = tgt.prefix4 AND s1.house = tgt.house` (with non-empty constraints).
- **Rule B**: `s1.country = tgt.country AND s1.business_address = tgt.business_address` (exact normalized address).
- **Rule C**: `s1.country = tgt.country AND s1.token1 = tgt.token1 AND s1.house = tgt.house`.
- **Rule D**: `s1.country = tgt.country AND TRIM(s1.business_name) = TRIM(tgt.business_name)`.

### Deduplication Logic
SQL `UNION` across the four rules, ensuring duplicate-free candidate pair generation.

### Outputs
- `P2/data/candidates/train_candidate_pairs_s2.tsv` (`source1_entity_id`, `matched_entity_id`, `label`)
- `P2/data/candidates/train_candidate_pairs_s3.tsv` (`source1_entity_id`, `matched_entity_id`, `label`)
- `P2/data/candidates/test_candidate_pairs_s2.tsv` (`source1_entity_id`, `matched_entity_id`)
- `P2/data/candidates/test_candidate_pairs_s3.tsv` (`source1_entity_id`, `matched_entity_id`)

---

## 3. Actual V2 Implementation

The canonical V2 generator is [P2/scripts/cands_BCD_v2_tasks.py](file:///Users/krishnagera/Amazon-ML-Challenge-2026/P2/scripts/cands_BCD_v2_tasks.py).

### Additional Derived Fields In-Query
```sql
regexp_replace(regexp_extract(TRIM(business_address), '[0-9]+[A-Za-z]?', 0), '^0+', '') AS house_norm,
regexp_replace(lower(trim(business_name)), '[^a-z0-9]', '', 'g') AS clean_name,
regexp_replace(lower(trim(business_address)), '[^a-z0-9]', '', 'g') AS clean_addr,
LEFT(TRIM(business_name), 3) AS prefix3,
CASE 
    WHEN lower(split_part(TRIM(business_name), ' ', 1)) IN ('the', 'shri', 'sri', 'dr', 'm/s', 'hotel', 'new', 'om', 'sai', 'jai', 'a', 'an')
         AND split_part(TRIM(business_name), ' ', 2) <> '' 
    THEN split_part(TRIM(business_name), ' ', 2)
    ELSE split_part(TRIM(business_name), ' ', 1)
END AS root_token,
regexp_extract(TRIM(business_address), '(?:^|[^0-9])([0-9]{5,6})(?:[^0-9]|$)', 1) AS zip_code
```

### Exact Blocking Rules
- **Rules A, B, C, D**: Baseline V1 rules.
- **Rule E**: `s1.country = tgt.country AND s1.clean_name = tgt.clean_name AND LENGTH(s1.clean_name) >= 3`.
- **Rule F1**: `s1.country = tgt.country AND s1.prefix4 = tgt.prefix4 AND s1.house_norm = tgt.house_norm`.
- **Rule F2**: `s1.country = tgt.country AND s1.token1 = tgt.token1 AND s1.house_norm = tgt.house_norm`.
- **Rule G**: `s1.country = tgt.country AND s1.root_token = tgt.root_token AND LENGTH(s1.root_token) >= 3 AND s1.house_norm = tgt.house_norm`.
- **Rule H**: `s1.country = tgt.country AND s1.clean_addr = tgt.clean_addr AND LENGTH(s1.clean_addr) >= 6`.
- **Rule I**: `s1.country = tgt.country AND s1.zip_code = tgt.zip_code AND LENGTH(s1.zip_code) >= 5 AND s1.prefix3 = tgt.prefix3 AND LENGTH(s1.prefix3) = 3`.

### Deduplication Logic
SQL `UNION` across all 9 queries.

### Outputs
- `P2/data/candidates/train_candidate_pairs_s2_v2.tsv`
- `P2/data/candidates/train_candidate_pairs_s3_v2.tsv`
- `P2/data/candidates/test_candidate_pairs_s2_v2.tsv`
- `P2/data/candidates/test_candidate_pairs_s3_v2.tsv`

---

## 4. Measured Candidate Metrics (Canonical Baseline)

All figures calculated via [P1/scripts/evaluation/evaluate_candidate_set.py](file:///Users/krishnagera/Amazon-ML-Challenge-2026/P1/scripts/evaluation/evaluate_candidate_set.py) against [train_ground_truth.tsv](file:///Users/krishnagera/Amazon-ML-Challenge-2026/data/train/train_ground_truth.tsv).

| Candidate Set | Target | Rows | Unique Rows | Dups | GT Total | Captured | Missed | Recall | Cand/GT Ratio |
|---|---|---|---|---|---|---|---|---|---|
| **V1 Train S2** | S2 | 24,594,064 | 24,594,064 | 0 | 3,693,619 | 2,214,137 | 1,479,482 | **59.9449%** | 6.659 |
| **V1 Train S3** | S3 | 29,998,661 | 29,998,661 | 0 | 3,944,746 | 2,346,442 | 1,598,304 | **59.4827%** | 7.605 |
| **V1 Train Combined** | S2+S3 | 54,592,725 | 54,592,725 | 0 | 7,638,365 | 4,560,579 | 3,077,786 | **59.7062%** | 7.147 |
| **V2 Train S2** | S2 | 30,359,040 | 30,359,040 | 0 | 3,693,619 | 2,438,575 | 1,255,044 | **66.0213%** | 8.219 |
| **V2 Train S3** | S3 | 36,973,484 | 36,973,484 | 0 | 3,944,746 | 2,584,593 | 1,360,153 | **65.5199%** | 9.373 |
| **V2 Train Combined** | S2+S3 | 67,332,524 | 67,332,524 | 0 | 7,638,365 | 5,023,168 | 2,615,197 | **65.7623%** | 8.815 |
| **V1 Test S2** | S2 | 30,232,352 | 30,232,352 | 0 | N/A | N/A | N/A | N/A | N/A |
| **V1 Test S3** | S3 | 35,822,910 | 35,822,910 | 0 | N/A | N/A | N/A | N/A | N/A |
| **V2 Test S2** | S2 | 34,919,169 | 34,919,169 | 0 | N/A | N/A | N/A | N/A | N/A |
| **V2 Test S3** | S3 | 41,714,627 | 41,714,627 | 0 | N/A | N/A | N/A | N/A | N/A |
| **Legacy Step 1 Test S2** | S2 | 26,046,195 | 26,046,195 | 0 | N/A | N/A | N/A | N/A | N/A |
| **Legacy Step 1 Test S3** | S3 | 30,777,878 | 30,777,878 | 0 | N/A | N/A | N/A | N/A | N/A |

---

## 5. Comparison Against Phase 0 Baseline

| Metric | Phase 0 Reported | Phase 2 Measured | Difference | Status |
|---|---|---|---|---|
| V1 S2 Recall | 59.9449% | 59.9449% | 0.0000% | **EXACT MATCH** |
| V1 S3 Recall | 59.4827% | 59.4827% | 0.0000% | **EXACT MATCH** |
| V1 Combined Recall | 59.7062% | 59.7062% | 0.0000% | **EXACT MATCH** |
| V2 S2 Recall | 66.0213% | 66.0213% | 0.0000% | **EXACT MATCH** |
| V2 S3 Recall | 65.5199% | 65.5199% | 0.0000% | **EXACT MATCH** |
| V2 Combined Recall | 65.7623% | 65.7623% | 0.0000% | **EXACT MATCH** |
| V1 S2 Row Count | 24,594,064 | 24,594,064 | 0 | **EXACT MATCH** |
| V1 S3 Row Count | 29,998,661 | 29,998,661 | 0 | **EXACT MATCH** |
| V2 S2 Row Count | 30,359,040 | 30,359,040 | 0 | **EXACT MATCH** |
| V2 S3 Row Count | 36,973,484 | 36,973,484 | 0 | **EXACT MATCH** |

**Conclusion**: Zero discrepancy. The candidate baselines are rock-solid and verified.

---

## 6. Artifact & Code Provenance Verification

Cryptographic verification demonstrates exact provenance between stored files and generator manifests:

1. **V2 Candidates**:
   - `train_candidate_pairs_s2_v2.tsv`: SHA256 `386c0a0c...` matches manifest SHA256.
   - `train_candidate_pairs_s3_v2.tsv`: SHA256 `6d6cc517...` matches manifest SHA256.
   - `test_candidate_pairs_s2_v2.tsv`: SHA256 `149e168a...` matches manifest SHA256.
   - `test_candidate_pairs_s3_v2.tsv`: SHA256 `ec2dd1c0...` matches manifest SHA256.
2. **V1 Candidates**:
   - `train_candidate_pairs_s2.tsv`: Sorted pair hash `b0babba4e35378f7b1f7e282625409de8513a5c88c4ef7605563188e8e5499cd` matches manifest hash.
   - `train_candidate_pairs_s3.tsv`: Sorted pair hash `011fb8228d2112c9d18d4e1756f903925ed9750033abac7aff6962a5ea9800ed` matches manifest hash.
   - `test_candidate_pairs_s2.tsv`: Sorted pair hash `5bc2b84efc6ff6644dfbba17559b46f274fdafebc810b5cf1dda797e3cf7c3f8` matches manifest hash.
   - `test_candidate_pairs_s3.tsv`: Sorted pair hash `89a053a9e30aafa9a2f37d1e49917636dfec55ba8c567db55f4bbc95c9004dbc` matches manifest hash.

**Provenance Verdict**: **PROVENANCE VERIFIED**.

---

## 7. Train / Test Retrieval Parity Audit

| Dataset / Generator | Target | Rules Executed | Input Files | Output Location | Parity with Train V1/V2 |
|---|---|---|---|---|---|
| **P2 V1 Train** (`cands_BCD_v1_tasks.py`) | S2, S3 | A, B, C, D | Normalized Train TSV | `P2/data/candidates/train_candidate_pairs_s*.tsv` | Baseline |
| **P2 V1 Test** (`cands_BCD_v1_tasks.py`) | S2, S3 | A, B, C, D | Normalized Test TSV | `P2/data/candidates/test_candidate_pairs_s*.tsv` | **FULL PARITY** with Train V1 |
| **P2 V2 Train** (`cands_BCD_v2_tasks.py`) | S2, S3 | A..I | Normalized Train TSV | `P2/data/candidates/train_candidate_pairs_s*_v2.tsv` | Baseline |
| **P2 V2 Test** (`cands_BCD_v2_tasks.py`) | S2, S3 | A..I | Normalized Test TSV | `P2/data/candidates/test_candidate_pairs_s*_v2.tsv` | **FULL PARITY** with Train V2 |
| **Legacy Test Step 1** (`03_blocking.ipynb`) | S2, S3 | Strategy B (A, B only) | Normalized Test TSV | `outputs/person1_step1/test_candidate_pairs_s*.tsv` | **DISCONNECTED** (Missing Rules C, D, E..I) |

### Impact of Legacy Pipeline Bug
Downstream scripts [finish_pipeline.py](file:///Users/krishnagera/Amazon-ML-Challenge-2026/P2/scripts/finish_pipeline.py) and [phase3_to_8_pipeline.py](file:///Users/krishnagera/Amazon-ML-Challenge-2026/P2/scripts/phase3_to_8_pipeline.py) mistakenly read from `outputs/person1_step1/test_candidate_pairs_s*.tsv` instead of `P2/data/candidates/test_candidate_pairs_s*.tsv`. This directly caused **8,872,974 test candidate pairs** (in S2) and **10,936,749 test candidate pairs** (in S3) to be dropped from scoring.

---

## 8. Candidate Fanout & Distribution Analysis

Candidate fanout per S1 entity was computed across the entire entity pool (2,206,821 S1 entities for train; 1,655,130 S1 entities for test):

| Candidate Set | Total S1 Pool | Zero Cand S1 | Zero Rate | Min | p50 (Median) | p90 | p95 | p99 | p99.9 | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| **V1 Train S2** | 2,206,821 | 346,953 | 15.72% | 0 | 2.0 | 30.0 | 55.0 | 124.0 | 323.0 | 1,322 |
| **V1 Train S3** | 2,206,821 | 325,444 | 14.75% | 0 | 3.0 | 38.0 | 68.0 | 166.0 | 442.0 | 1,650 |
| **V1 Train Combined** | 2,206,821 | 124,148 | 5.63% | 0 | 5.0 | 63.0 | 123.0 | 284.0 | 721.0 | 2,972 |
| **V2 Train S2** | 2,206,821 | 277,448 | 12.57% | 0 | 3.0 | 38.0 | 70.0 | 154.0 | 373.0 | 1,477 |
| **V2 Train S3** | 2,206,821 | 262,072 | 11.88% | 0 | 3.0 | 48.0 | 85.0 | 199.0 | 511.0 | 1,829 |
| **V2 Train Combined** | 2,206,821 | 94,043 | 4.26% | 0 | 5.0 | 81.0 | 155.0 | 346.0 | 850.0 | 3,306 |
| **V1 Test S2** | 1,655,130 | 254,483 | 15.38% | 0 | 3.0 | 45.0 | 96.0 | 219.0 | 408.0 | 1,510 |
| **V1 Test S3** | 1,655,130 | 239,473 | 14.47% | 0 | 3.0 | 56.0 | 122.0 | 243.0 | 503.0 | 1,820 |
| **V2 Test S2** | 1,655,130 | 206,207 | 12.46% | 0 | 3.0 | 55.0 | 111.0 | 239.0 | 488.0 | 1,703 |
| **V2 Test S3** | 1,655,130 | 195,392 | 11.81% | 0 | 3.0 | 67.0 | 142.0 | 277.0 | 582.0 | 2,004 |
| **Legacy Step 1 Test S2** | 1,655,130 | 400,630 | **24.21%** | 0 | 2.0 | 37.0 | 90.0 | 215.0 | 401.0 | 1,510 |
| **Legacy Step 1 Test S3** | 1,655,130 | 381,967 | **23.08%** | 0 | 2.0 | 47.0 | 112.0 | 232.0 | 498.0 | 1,820 |

### Key Takeaway on Fanout
- V2 successfully recovered 30,105 S1 entities that had zero candidates under V1 in training.
- Max fanout grew moderately from 2,972 to 3,306 pairs for an S1 entity.
- The 95th percentile grew from 123 to 155 candidates per S1 entity in Combined Train.

---

## 9. Data Integrity & Schema Validation Results

Inspected across all 10 candidate sets:
- **Duplicate rows**: 0 (0.000%)
- **Null IDs**: 0 (0.000%)
- **Self-matches**: 0 (0.000%)
- **Malformed IDs**: 0 (0.000%)
- **Invalid IDs (not in canonical entity tables)**: 0 (0.000%)
- **Cross-country candidates**: 0 (0.000%)

All 10 candidate sets achieved a clean **PASS** on schema and referential integrity.

---

## 10. Recommendations for Phase 3 Retrieval Experiments

1. **Mandatory Candidate Input**: In Phase 3, all candidate additions must build directly from the verified P1 canonical entities ([P1/data/entities/](file:///Users/krishnagera/Amazon-ML-Challenge-2026/P1/data/entities/)).
2. **Evaluator Standardization**: Every experiment must be evaluated using [P1/scripts/evaluation/evaluate_candidate_set.py](file:///Users/krishnagera/Amazon-ML-Challenge-2026/P1/scripts/evaluation/evaluate_candidate_set.py) to measure GT capture, missed count, exact recall, and fanout quantiles.
3. **Train / Test Dual Generation**: Any rule added to training candidates must be simultaneously applied to test candidate generation to preserve strict train/test parity.
4. **Fanout Guardrails**: Candidate expansion in Phase 3 must monitor the 99th percentile and max fanout to prevent combinatorial explosion.
