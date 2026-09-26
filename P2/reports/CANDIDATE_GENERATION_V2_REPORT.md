# Candidate Generation V2 Report

## 1. Executive Summary
This report documents the design, diagnostic analysis, implementation, and empirical validation of candidate generation version 2 (`cands_BCD_v2`) for the Amazon ML Challenge 2026 Entity Resolution pipeline.

The frozen canonical baseline (`cands_BCD_v1`) captured **4,560,579 true pairs** out of 7,638,365 ground-truth pairs, yielding a candidate recall ceiling of **59.706%** across 54,592,725 candidate pairs. Through systematic SQL anti-join diagnostic profiling of all 3,077,786 missed ground-truth pairs, we identified specific structural failure modes in V1:
1. Punctuation and formatting variance in business names (e.g., corporate designators like `inc.` vs `inc`, hyphenated entities, special annotations).
2. Formatted numerical prefixes in addresses containing leading zeros (e.g., `00622` vs `622`).
3. Common non-distinctive leading tokens (e.g., `the`, `shri`, `sri`, `dr`, `m/s`, `hotel`, `new`) masking the true discriminative business name root.
4. Clean alphanumeric address equivalence where whitespace/punctuation varied.
5. High-precision postal code (ZIP/PIN) locality clustering with name prefix matching.

By introducing five controlled, high-precision blocking rules, `cands_BCD_v2` increases combined candidate recall from **59.706% to 65.762% (+6.056 percentage points)**, capturing **+462,589 additional true entity pairs** while strictly bounding candidate volume to a **1.233x multiplier** (67,332,524 pairs vs 54,592,725 pairs). Every single baseline artifact remains 100% frozen, untouched, and preserved.

---

## 2. Repository / Environment
- **Repository Root**: `/Users/krishnagera/Amazon-ML-Challenge-2026`
- **Git Branch**: `main`
- **Starting Git Commit**: `5cfbaa1db957485f7bb33a4e7b9a8d732256d8a7`
- **Operating System**: macOS (Darwin 24.3.0, arm64)
- **Python Version**: `3.13.5 (packaged by Anaconda, Inc., Jun 12 2025)`
- **DuckDB Version**: `1.5.5`
- **Pandas Version**: `3.0.5`
- **NumPy Version**: `2.5.2`

---

## 3. Input Dataset Verification
All normalized source files located in `outputs/person1_step1/normalized/` were directly inspected and validated. Each file possesses exactly 4 columns: `entity_id`, `business_name`, `business_address`, `country`. No dedicated fields for `postal_code`, `city`, `state`, or `street` exist in the schema; any sub-component logic must be derived via deterministic string extraction.

| Dataset File | Total Rows | Columns | Empty Entity IDs | Duplicate IDs | Missing Names | Missing Addresses | Duplicate Names | Duplicate Addrs | Country Breakdown |
|---|---|---|---|---|---|---|---|---|---|
| `train_source1_normalized.tsv` | 2,206,821 | 4 | 0 | 0 | 0 | 0 | 669,619 | 76,215 | US: 1,323,633<br>INDIA: 883,188 |
| `train_source2_normalized.tsv` | 5,034,616 | 4 | 0 | 0 | 6 | 168,967 | 882,120 | 565,567 | US: 3,016,817<br>INDIA: 2,017,799 |
| `train_source3_normalized.tsv` | 5,285,603 | 4 | 0 | 0 | 18 | 175,916 | 857,208 | 478,593 | US: 3,170,056<br>INDIA: 2,115,547 |
| `test_source1_normalized.tsv` | 1,732,544 | 4 | 0 | 0 | 0 | 0 | 495,384 | 58,006 | INDIA: 809,986<br>US: 663,106<br>FRANCE: 259,452 |
| `test_source2_normalized.tsv` | 4,887,273 | 4 | 0 | 0 | 49 | 129,408 | 787,877 | 575,852 | INDIA: 2,312,565<br>US: 1,871,330<br>FRANCE: 703,378 |
| `test_source3_normalized.tsv` | 5,082,316 | 4 | 0 | 0 | 61 | 136,098 | 758,959 | 496,158 | INDIA: 2,405,000<br>US: 1,945,701<br>FRANCE: 731,615 |

---

## 4. Ground Truth Verification
Ground truth was verified directly from `outputs/person1_step1/train_ground_truth_reconstructed.tsv`.
- **Exact Path**: `/Users/krishnagera/Amazon-ML-Challenge-2026/outputs/person1_step1/train_ground_truth_reconstructed.tsv`
- **Total Rows**: 2,206,821 (matches Source 1 entity count 1-to-1)
- **Schema**: `source1_entity_id VARCHAR`, `matched_entity_ids VARCHAR`
- **Missing S1 IDs**: 0
- **Missing / Blank Matched Entity IDs**: 123,247 (these are true negative S1 queries with 0 ground-truth matches, representing the theoretical candidate floor of `123,247 / 2,206,821 = 5.5848%`)
- **Distinct S1 Entities with True Matches**: 2,083,574
- **Total Expanded True Pairs**: 7,638,365
- **Duplicate Pairs**: 0
- **True S1 → S2 Pairs**: 3,693,619 (48.36%)
- **True S1 → S3 Pairs**: 3,944,746 (51.64%)
- **True Pairs pointing to other sources**: 0

---

## 5. V1 Baseline
The baseline candidate generation rules (`cands_BCD_v1`) defined in `P2/scripts/cands_BCD_v1_tasks.py` were independently executed and verified against the canonical report artifacts:
- **Rule A**: `s1.country = tgt.country AND prefix4(name) = prefix4(name) AND house = house`
- **Rule B**: `s1.country = tgt.country AND business_address = business_address`
- **Rule C**: `s1.country = tgt.country AND token1(name) = token1(name) AND house = house`
- **Rule D**: `s1.country = tgt.country AND business_name = business_name`

### Verified Recomputed Baseline Table
| Metric | V1 S2 | V1 S3 | V1 Combined |
|---|---|---|---|
| Ground-Truth Pairs | 3,693,619 | 3,944,746 | 7,638,365 |
| Candidate Pairs | 24,594,064 | 29,998,661 | 54,592,725 |
| Captured True Pairs | 2,214,137 | 2,346,442 | 4,560,579 |
| Candidate Recall | 59.9449% | 59.4827% | 59.7062% |
| Missed True Pairs | 1,479,482 | 1,598,304 | 3,077,786 |
| Candidate / GT Ratio | 6.6585x | 7.6047x | 7.1472x |
| Distinct S1 with Cands | 1,859,868 | 1,881,377 | - |
| S1 with 0 Candidates | 346,953 | 325,444 | - |

The sorted pairs SHA-256 hashes generated by our run match `P2/reports/cands_BCD_v1_manifest.tsv` byte-for-byte:
- `train_candidate_pairs_s2.tsv`: `b0babba4e35378f7b1f7e282625409de8513a5c88c4ef7605563188e8e5499cd`
- `train_candidate_pairs_s3.tsv`: `011fb8228d2112c9d18d4e1756f903925ed9750033abac7aff6962a5ea9800ed`

---

## 6. V1 Missed-Pair Analysis
Using DuckDB anti-joins:
```sql
SELECT gt.* FROM gt_{src} gt
WHERE NOT EXISTS (SELECT 1 FROM v1_{src} c WHERE c.source1_entity_id = gt.source1_entity_id AND c.matched_entity_id = gt.matched_entity_id)
```
We isolated exactly:
- **1,479,482** missed S2 true pairs
- **1,598,304** missed S3 true pairs
- **3,077,786** total missed pairs

Every missed pair was profiled across character n-grams, token sets, house number matches, clean alphanumeric representations, and postal codes.

---

## 7. Root Causes of Missed Pairs
Empirical SQL breakdown across the 3,077,786 missed pairs revealed five definitive root causes:
1. **100% Country Concordance**: 0 missed pairs had conflicting countries or blank countries. Country partitioning is flawless and optimal.
2. **Punctuation and Normalization Gaps in Business Name**:
   - `exact_clean_name`: **168,471 missed true pairs** (77,679 in S2, 90,792 in S3) have 100% identical business names once punctuation and whitespace are stripped (e.g., `sky technology pvt ltd` vs `sky technology pvt-ltd`, `dr narayan associates` vs `dr narayan associates ((company))`). Rule D failed solely due to punctuation.
3. **Leading Zeros in House Numbers**:
   - Over **164,093 true pairs** were blocked from matching Rule A or Rule C because one source contained leading zeros (e.g., `00622` vs `622`, `00327` vs `327`). Stripping leading zeros (`regexp_replace(house, '^0+', '')`) re-aligns these numbers.
4. **Leading Stopwords & Titles Masking Discriminative Token**:
   - In **256,763 true pairs**, the true entity name was shifted to token 2 due to common prefixes (`the`, `shri`, `sri`, `dr`, `m/s`, `hotel`, `new`, `om`, `sai`, `jai`). Rule C (token1) and Rule A (prefix4) missed them because `token1` was matched against the prefix rather than the root entity.
5. **Exact Clean Address Equivalence**:
   - **17,075 missed pairs** possessed identical alphanumeric addresses (`clean_addr`) differing only in formatting (e.g., `2/18` vs `no 2/18-`).
6. **Non-ASCII Indic Script Variations**:
   - In **693,416 missed pairs** (386,517 in S2, 306,899 in S3), the target business name was recorded in an Indic script (Devanagari, Telugu, Odia, Bengali, Malayalam) while Source 1 was Latin script. These pairs rely entirely on address/locality/house number matching.

---

## 8. Candidate Rules Investigated
We evaluated 9 candidate generation strategies designed to address the observed failure modes:
- **Rule E (`Rule_CleanName`)**: Country + alphanumeric clean name (`LENGTH >= 3`).
- **Rule F1 (`Rule_HouseNorm_A`)**: Country + `prefix4(name)` + zero-stripped house number.
- **Rule F2 (`Rule_HouseNorm_C`)**: Country + `token1(name)` + zero-stripped house number.
- **Rule G (`Rule_RootToken_House`)**: Country + semantic root token (skipping leading titles) + zero-stripped house number.
- **Rule H (`Rule_CleanAddress`)**: Country + clean alphanumeric address (`LENGTH >= 6`).
- **Rule I (`Rule_Zip_Prefix3`)**: Country + 5/6-digit postal code + `prefix3(name)`.
- **Rule J (`Rule_Zip_Token1`)**: Country + postal code + `token1(name)`.
- **Rule K (`Rule_Prefix3_House`)**: Country + `prefix3(name)` + zero-stripped house.
- **Rule L (`Rule_Token2_House`)**: Country + `token2(name)` + zero-stripped house.

---

## 9. Rule-Level Measurements

### Independent Evaluation on S2 (GT = 3,693,619 | V1 Captured = 2,214,137)
| Rule | Candidate Count | Positives Captured | Marginal Pos vs V1 | Marginal Recall | Precision Proxy | Max Cands / S1 | Decision |
|---|---|---|---|---|---|---|---|
| `Rule_CleanName` | 10,849,350 | 836,817 | +77,679 | +2.103% | 0.0771 | 612 | **ADOPTED** |
| `Rule_HouseNorm_A` | 18,195,217 | 2,005,683 | +82,807 | +2.242% | 0.1102 | 1,477 | **ADOPTED** |
| `Rule_HouseNorm_C` | 12,939,469 | 1,844,760 | +74,033 | +2.004% | 0.1426 | 1,008 | **ADOPTED** |
| `Rule_RootToken_House` | 12,995,701 | 1,808,933 | +121,126 | +3.279% | 0.1392 | 476 | **ADOPTED** |
| `Rule_CleanAddress` | 570,598 | 464,898 | +16,804 | +0.455% | 0.8148 | 19 | **ADOPTED** |
| `Rule_Zip_Prefix3` | 166,478 | 158,412 | +6,030 | +0.163% | 0.9515 | 8 | **ADOPTED** |
| `Rule_Zip_Token1` | 140,926 | 137,153 | +4,334 | +0.117% | 0.9732 | 8 | REDUNDANT (Subsumed by Prefix3) |
| `Rule_Prefix3_House` | 31,572,848 | 2,035,833 | +99,748 | +2.701% | 0.0645 | 3,124 | **REJECTED** (+15M cands volume explosion) |
| `Rule_Token2_House` | 35,748,524 | 1,209,561 | +64,336 | +1.742% | 0.0338 | 1,200 | **REJECTED** (+34M cands volume explosion) |

### Independent Evaluation on S3 (GT = 3,944,746 | V1 Captured = 2,346,442)
| Rule | Candidate Count | Positives Captured | Marginal Pos vs V1 | Marginal Recall | Precision Proxy | Max Cands / S1 | Decision |
|---|---|---|---|---|---|---|---|
| `Rule_CleanName` | 11,892,958 | 915,764 | +90,792 | +2.302% | 0.0770 | 588 | **ADOPTED** |
| `Rule_HouseNorm_A` | 23,287,825 | 2,072,253 | +81,286 | +2.061% | 0.0890 | 1,829 | **ADOPTED** |
| `Rule_HouseNorm_C` | 17,022,318 | 1,904,425 | +72,254 | +1.832% | 0.1119 | 1,126 | **ADOPTED** |
| `Rule_RootToken_House` | 17,112,164 | 1,876,916 | +135,637 | +3.438% | 0.1097 | 439 | **ADOPTED** |
| `Rule_CleanAddress` | 206,735 | 171,054 | +271 | +0.007% | 0.8274 | 6 | **ADOPTED** |
| `Rule_Zip_Prefix3` | 169,151 | 160,653 | +7,039 | +0.178% | 0.9498 | 9 | **ADOPTED** |
| `Rule_Token2_House` | 41,803,744 | 1,220,312 | +67,588 | +1.713% | 0.0292 | 1,244 | **REJECTED** (+40M cands volume explosion) |

---

## 10. Incremental Ablation
The cumulative impact of adding each adopted rule sequentially was computed and recorded in `P2/reports/cands_BCD_v2_rule_ablation.tsv`.

### Source 2 Cumulative Progression
| Step | Rule Added | Cumulative Candidates | Marginal Candidates | Cumulative Positives | Marginal Positives | Cumulative Recall | Candidate / GT Ratio |
|---|---|---|---|---|---|---|---|
| 0 | V1 Baseline (A+B+C+D) | 24,594,064 | 24,594,064 | 2,214,137 | 2,214,137 | 59.9449% | 6.6585x |
| 1 | + CleanName | 27,048,631 | +2,454,567 | 2,291,816 | +77,679 | 62.0480% | 7.3231x |
| 2 | + HouseNorm (A & C) | 28,967,319 | +1,918,688 | 2,367,469 | +75,653 | 64.0962% | 7.8425x |
| 3 | + RootToken_House | 30,318,969 | +1,351,650 | 2,418,375 | +50,906 | 65.4744% | 8.2085x |
| 4 | + CleanAddress | 30,350,288 | +31,319 | 2,433,218 | +14,843 | 65.8763% | 8.2170x |
| 5 | + Zip_Prefix3 | 30,359,040 | +8,752 | 2,438,575 | +5,357 | **66.0213%** | **8.2193x** |

### Source 3 Cumulative Progression
| Step | Rule Added | Cumulative Candidates | Marginal Candidates | Cumulative Positives | Marginal Positives | Cumulative Recall | Candidate / GT Ratio |
|---|---|---|---|---|---|---|---|
| 0 | V1 Baseline (A+B+C+D) | 29,998,661 | 29,998,661 | 2,346,442 | 2,346,442 | 59.4827% | 7.6047x |
| 1 | + CleanName | 32,727,411 | +2,728,750 | 2,437,234 | +90,792 | 61.7843% | 8.2965x |
| 2 | + HouseNorm (A & C) | 35,209,907 | +2,482,496 | 2,510,989 | +73,755 | 63.6540% | 8.9258x |
| 3 | + RootToken_House | 36,962,455 | +1,752,548 | 2,578,113 | +67,124 | 65.3556% | 9.3700x |
| 4 | + CleanAddress | 36,963,419 | +964 | 2,578,349 | +236 | 65.3616% | 9.3703x |
| 5 | + Zip_Prefix3 | 36,973,484 | +10,065 | 2,584,593 | +6,244 | **65.5199%** | **9.3728x** |

---

## 11. Final V2 Blocking Strategy
The final V2 strategy combines:
1. All 4 canonical V1 baseline rules (Rule A, Rule B, Rule C, Rule D).
2. **Rule E (Clean Name)**: `country = country AND clean_name = clean_name (len >= 3)`.
3. **Rule F (Normalized House A & C)**: `country = country AND house_norm = house_norm AND (prefix4 = prefix4 OR token1 = token1)`.
4. **Rule G (Semantic Root Token)**: `country = country AND house_norm = house_norm AND root_token = root_token (len >= 3)`.
5. **Rule H (Clean Address)**: `country = country AND clean_addr = clean_addr (len >= 6)`.
6. **Rule I (Postal Code Locality)**: `country = country AND zip_code = zip_code (len >= 5) AND prefix3 = prefix3 (len = 3)`.

All rules are executed via unified `UNION` projection in DuckDB, deduplicating pairs natively in a single pass.

---

## 12. V1 vs V2 Metrics

| Metric | V1 S2 | V2 S2 | V1 S3 | V2 S3 | V1 Combined | V2 Combined | Delta (Combined) |
|---|---|---|---|---|---|---|---|
| **Ground-Truth Pairs** | 3,693,619 | 3,693,619 | 3,944,746 | 3,944,746 | 7,638,365 | 7,638,365 | 0 |
| **Candidate Pairs** | 24,594,064 | 30,359,040 | 29,998,661 | 36,973,484 | 54,592,725 | 67,332,524 | +12,739,799 (+23.33%) |
| **Captured True Pairs** | 2,214,137 | 2,438,575 | 2,346,442 | 2,584,593 | 4,560,579 | 5,023,168 | **+462,589** |
| **Candidate Recall** | 59.9449% | **66.0213%** | 59.4827% | **65.5199%** | 59.7062% | **65.7623%** | **+6.0561%** |
| **Missed True Pairs** | 1,479,482 | 1,255,044 | 1,598,304 | 1,360,153 | 3,077,786 | 2,615,197 | **-462,589 (-15.03%)** |
| **Candidate / GT Ratio** | 6.6585x | 8.2193x | 7.6047x | 9.3728x | 7.1472x | 8.8150x | +1.6678x |

---

## 13. Candidate Volume Analysis
Candidate generation volume must balance high recall against downstream modeling inference costs.
- **Candidate Volume Multiplier**: 1.233x (Volume expanded from 54.59M to 67.33M candidates).
- **Candidate Precision Proxy**: `5,023,168 / 67,332,524 = 7.460%` (Baseline was `4,560,579 / 54,592,725 = 8.354%`).
- **S1 Coverage**:
  - Distinct S1 with candidates in S2: 1,859,868 (V1) → **1,929,373 (V2)** (+69,505 S1 entities covered).
  - Distinct S1 with candidates in S3: 1,881,377 (V1) → **1,944,749 (V2)** (+63,372 S1 entities covered).
- **Per-Entity Candidate Distributions**:
  - `p50`: 5 candidates per S1
  - `p90`: 23 candidates per S1
  - `p95`: 41 candidates per S1
  - `p99`: 118 candidates per S1
  - Extreme tail keys were bounded by rejecting generic 2-token joins and unbound prefix3 joins.

---

## 14. Integrity Validation
All generated V2 candidate artifacts underwent automated programmatic integrity auditing:
- **Missing S1 IDs**: 0 across all 4 files.
- **Missing Target IDs**: 0 across all 4 files.
- **Foreign / Invalid IDs**: 0 IDs outside of the respective normalized dataset tables.
- **Duplicate Pairs**: Exactly 0 duplicate `(source1_entity_id, matched_entity_id)` pairs in all 4 files.
- **Target Source Isolation**: Exactly 0 cross-contamination (100% of S2 candidate IDs start with `S2-`; 100% of S3 candidate IDs start with `S3-`).
- **Training Labels**: Every label in `train_candidate_pairs_s2_v2.tsv` and `train_candidate_pairs_s3_v2.tsv` is strictly binary (`0` or `1`) and derived 100% deterministically from `train_ground_truth_reconstructed.tsv`. Label mismatch count vs ground truth is exactly 0.
- **Test Candidates**: Exact schema `source1_entity_id`, `matched_entity_id` with 0 fake labels or inferences.

---

## 15. Artifact Manifest

| Artifact File | Split / Source | Row Count | File Size (Bytes) | File SHA-256 |
|---|---|---|---|---|
| `train_candidate_pairs_s2_v2.tsv` | Train S2 | 30,359,040 | 843,272,201 | `386c0a0c0dacda2df90d26778a893c760b686c1c4e84bc0ef033e7ce84c28d22` |
| `train_candidate_pairs_s3_v2.tsv` | Train S3 | 36,973,484 | 1,027,044,644 | `6d6cc5171beb4f445a4016f35a27e488e543cdeaa66bb5a26b342f87302c8404` |
| `test_candidate_pairs_s2_v2.tsv` | Test S2 | 34,919,169 | 900,172,941 | `149e168a56025a860c7e917f1ba70f990cfaefd0660175d8d8050c657fbe9c82` |
| `test_candidate_pairs_s3_v2.tsv` | Test S3 | 41,714,627 | 1,075,341,054 | `ec2dd1c0ca63d71d242b84854b22bace3fec0c7d611a193fc224a6b6bf971576` |
| `cands_BCD_v2_manifest.tsv` | Manifest | 5 | 1,889 | `a9062fdba12c7dcaee9524ba17a022f42a66eec285317f0e698bc43144a2c53b` |
| `cands_BCD_v2_rule_ablation.tsv` | Report | 13 | 1,268 | `2656372d8a4ba5bb94723fa55f2420fe813bf69b615d554a737f90378b871926` |
| `cands_BCD_v2_missed_pair_analysis.tsv` | Report | 3 | 352 | `3c81e3a6c9e01d18f596350f269a8449bf3ba89ef9446dbe649f783307612f02` |
| `cands_BCD_v2_extractor_spec.md` | Spec | 40 | 1,942 | `d829141be408b0be148ec7ee6d4a362544fe2822a15f0eb5e84a2ca98efb0649` |

All baseline V1 artifacts (`P2/data/candidates/*_s2.tsv`, `*_s3.tsv`, `E02_RUN.json`, `E02_MODELING_REPORT.md`) are completely preserved and untouched.

---

## 16. Reproduction Instructions
To reproduce the complete V2 candidate dataset and manifests from scratch, run:
```bash
python P2/scripts/cands_BCD_v2_tasks.py
```
- Dependencies: `duckdb >= 1.0.0`
- Runtime: ~75 seconds on Apple M-series / 4-core machine.
- Temporary disk space: ~5 GB in `P2/data/duckdb_tmp_v2_gen`.

---

## 17. Person 2 Modeling Handoff
### Frozen Baseline Context
- Frozen V1 candidate paths:
  - `P2/data/candidates/train_candidate_pairs_s2.tsv`
  - `P2/data/candidates/train_candidate_pairs_s3.tsv`
  - `P2/data/candidates/test_candidate_pairs_s2.tsv`
  - `P2/data/candidates/test_candidate_pairs_s3.tsv`
- Canonical baseline model report: `P2/reports/E02_MODELING_REPORT.md` (Fixed `T=0.5`, Scorer Macro F0.5: `0.7695`).

### New V2 Artifacts Ready for Person 2
- **Training Candidates**:
  - `P2/data/candidates/train_candidate_pairs_s2_v2.tsv` (30,359,040 rows, schema: `source1_entity_id`, `matched_entity_id`, `label`)
  - `P2/data/candidates/train_candidate_pairs_s3_v2.tsv` (36,973,484 rows, schema: `source1_entity_id`, `matched_entity_id`, `label`)
- **Test Candidates**:
  - `P2/data/candidates/test_candidate_pairs_s2_v2.tsv` (34,919,169 rows, schema: `source1_entity_id`, `matched_entity_id`)
  - `P2/data/candidates/test_candidate_pairs_s3_v2.tsv` (41,714,627 rows, schema: `source1_entity_id`, `matched_entity_id`)

### Feature Generation Note
- Feature files for V2 have **not** been generated.
- **Explicit Instruction**: Person 2 should generate model features from the V2 candidate files using the existing canonical feature pipeline (`P2/scripts/regenerate_features.py` or equivalent DuckDB SQL feature extractors).

### Action Items for Person 2
1. Verify the V2 candidate file row counts and SHA-256 hashes against `P2/reports/cands_BCD_v2_manifest.tsv`.
2. Verify candidate recall independently (`66.02%` S2, `65.52%` S3).
3. Generate feature tables using the canonical 18-feature SQL schema (`P2/data/features/cands_BCD_v2_features_train_s2.tsv`, etc.).
4. Train the 5-fold cross-validation LightGBM model on the V2 universe (using the existing 10% negative downsampling strategy established in E02).
5. Generate out-of-fold (OOF) predictions and test predictions.
6. Compare performance against the frozen E02 baseline (`0.7695`). **Do not overwrite `E02_MODELING_REPORT.md` or `E02_RUN.json`**; create a dedicated `E03_MODELING_REPORT.md`.

---

## 18. Person 3 Evaluation Handoff
### Upstream Prediction Format Requirements
When Person 2 hands predictions to Person 3, the prediction artifacts must contain:
- `source1_entity_id`: Source 1 entity identifier
- `matched_entity_id`: Target candidate identifier
- `score` or `probability`: Continuous model output score
- `predicted_label` or `match`: Binary decision at the chosen threshold

Person 3 must **not** accept an artifact containing only final matched IDs; full traceability from candidate entity → model score → threshold decision is required for evaluation across folds.

### Upstream Candidate Lineage
- Person 3 validation scorer must evaluate against candidates generated strictly from `P2/data/candidates/test_candidate_pairs_s2_v2.tsv` and `test_candidate_pairs_s3_v2.tsv`.
- Under no circumstances should V1 candidates and V2 model predictions be combined.

---

## 19. Limitations
1. **Transliterated Indic Scripts**: Approximately 693,416 remaining missed pairs (~26% of residual missed pairs) involve targets whose business names are written in Indic scripts while Source 1 is in Latin script. Without cross-lingual phonetic transliteration or embedding-based cross-script retrieval, string-based name blocking cannot match these entities.
2. **Missing House Numbers**: Approximately 152,877 remaining missed pairs lack house numbers in both records or in one record, forcing dependence on high-precision address/zip tokens.
3. **Severe Typographic Corruption**: In pairs with Jaro-Winkler string similarity < 0.60 on both names and addresses, deterministic key blocking reaches its practical precision limit.

---

## 20. Recommended Next Steps
1. **Cross-Script Transliteration**: Implement an unidecode/Indic transliteration preprocessor to map Devanagari, Telugu, Odia, and Bengali names to standard ASCII phonetic representations.
2. **Dense Neural Embeddings**: For entities with zero candidates or non-ASCII scripts, explore pre-trained bilingual multilingual-E5 or MiniLM vector blocking restricted strictly to within-country partitions.
3. **Person 2 Feature Engineering**: Person 2 should incorporate token-set intersection count and clean-name match indicators into the LightGBM feature suite to fully exploit the new candidate attributes.
