# Phase 1 — Canonical Entity Dataset & Missed-Pair Forensic Report

**Date**: 2026-09-26  
**Module**: Person 1 (P1)  
**Scope**: Canonical entity layer construction, lossless dataset validation, and deep forensic analysis of 2,615,197 V2 missed ground truth pairs  
**Phase Status**: COMPLETED  

---

## 1. Objective

Phase 1 executed two non-negotiable mandates:
1. **Objective A — Build Canonical Entity Dataset Layer**: Construct a standardized, lossless, deterministic Parquet representation for all six partitions (train S1/S2/S3, test S1/S2/S3), preserving 100% of raw attributes side-by-side with normalized text and deterministic blocking keys.
2. **Objective B — Forensic Analysis of V2 Missed Pairs**: Isolate and analyze all 2,615,197 true ground truth matches missed by the V2 candidate universe to determine mathematically why each pair failed and which retrieval channels are empirically justified for Phase 3.

*Rule of Engagement*: Zero pipeline optimizations, zero V3 generation, zero vector embeddings, and zero model modifications were introduced during this phase.

---

## 2. Phase 0 Baseline Reference

Phase 0 established and froze the verified repository baseline (Git commit `6653ec0`):
- **E02 Model Baseline OOF Macro $F_{0.5}$**: `0.769452` ($T=0.5$ on V1 candidates)
- **V1 Candidate Oracle Macro $F_{0.5}$**: `0.784523`
- **V1 Candidate Recall**: S2 = `59.9449%` (2,214,137 / 3,693,619) | S3 = `59.4827%` (2,346,442 / 3,944,746)
- **V2 Candidate Recall (Unused by E02)**: S2 = `66.0213%` (2,438,575 / 3,693,619) | S3 = `65.5199%` (2,584,593 / 3,944,746)
- **Total Ground Truth Pairs**: 7,638,365 pairs (3,693,619 S2 + 3,944,746 S3)
- **V2 Missed Ground Truth Pairs**: **2,615,197 pairs** (S2: 1,255,044; S3: 1,360,153)

---

## 3. Canonical Entity Dataset

Six canonical Parquet files were constructed under `P1/data/entities/` using DuckDB with ZSTD compression:
- `train/source1/train_s1_entities.parquet` (2,206,821 rows, 365.65 MB)
- `train/source2/train_s2_entities.parquet` (5,034,616 rows, 863.84 MB)
- `train/source3/train_s3_entities.parquet` (5,285,603 rows, 899.22 MB)
- `test/source1/test_s1_entities.parquet` (1,732,544 rows, 292.95 MB)
- `test/source2/test_s2_entities.parquet` (4,887,273 rows, 862.86 MB)
- `test/source3/test_s3_entities.parquet` (5,082,316 rows, 877.38 MB)

Total entity volume: **24,228,873 rows** (4,161.9 MB total on disk).

---

## 4. Schema

The canonical schema establishes 22 strictly typed columns:
- **Identifier**: `entity_id` (Primary Key, sorted ascending).
- **Raw Features**: `business_name_raw`, `business_address_raw`, `country_raw` (unmodified source values).
- **Normalized Features**: `business_name_normalized`, `business_address_normalized`, `country_normalized`.
- **Cleaned Strings**: `name_clean`, `address_clean`, `name_alnum`, `address_alnum`.
- **Token Lists**: `name_tokens`, `address_tokens` (`VARCHAR[]`).
- **Blocking Keys**: `name_prefix_1`, `name_prefix_2`, `name_prefix_3`, `name_prefix_4`, `first_token`, `root_token` (prefix-stripped), `house_number`, `house_number_norm` (zero-stripped), `postal_code` (5-6 digits).

Full specifications and real data examples are cataloged in `P1/reports/CANONICAL_ENTITY_SCHEMA.md`.

---

## 5. Row and ID Preservation

Independent validation performed via `validate_entity_datasets.py` proved:
- **Raw Rows vs. Canonical Rows**: Exact 100.0% retention across all 6 files. Zero rows dropped.
- **Unique Entity IDs**: Exactly matches raw counts (e.g., 2,206,821 for train S1).
- **Duplicate IDs**: Exactly 0 across all files.
- **Missing / Extra IDs**: Exactly 0. Set difference `raw - canonical = 0` and `canonical - raw = 0`.

Validation results are documented in `P1/reports/PHASE1_ENTITY_DATASET_VALIDATION.md` and `P1/manifests/entity_dataset_manifest.tsv`.

---

## 6. Determinism

All canonical Parquet datasets were written with strict `ORDER BY raw.entity_id ASC`.  
Validation confirmed that for 100% of rows, `entity_id` is monotonically increasing (`is_sorted = True`), guaranteeing bitwise reproducibility of all downstream blocking and feature operations.

---

## 7. V2 Missed-Pair Dataset

The complete set of true matches missed by V2 was extracted and verified:
- **`missed_s2.tsv`**: 1,255,044 rows (463 MB; SHA256: `3bcbef2a...`)
- **`missed_s3.tsv`**: 1,360,153 rows (486 MB; SHA256: `8669f6d8...`)
- **Total Missed Pairs**: **2,615,197 rows**
- **Integrity**:
  - In GT: 100% (2,615,197 / 2,615,197)
  - In V2 candidates: 0% (0 / 2,615,197)
  - Duplicate pairs: 0

---

## 8. Missed-Pair Country Analysis

Cataloged in `P1/reports/V2_MISSED_PAIR_COUNTRY_BREAKDOWN.tsv`:

| Target Source | Country | Missed Pair Count | Pct of Target Missed | Pct of Total Missed |
|---|---|---|---|---|
| `source2` | India | 700,504 | 55.82% | 26.79% |
| `source2` | US | 554,540 | 44.18% | 21.20% |
| `source3` | India | 732,109 | 53.83% | 27.99% |
| `source3` | US | 628,044 | 46.17% | 24.02% |

**Key Finding**:
- **54.78% of all missed pairs are located in India** (1,432,613 pairs).
- **45.22% are in the US** (1,182,584 pairs).
- **Cross-Country Mismatch in GT**: Exactly **0 pairs**. In 100% of ground truth matches, `country_s1 == country_target`. Country filtering is therefore completely safe and does not cause recall loss.

---

## 9. Name Lexical Variation Analysis

Jaro-Winkler similarity percentiles computed across all missed pairs:

| Target Source | Min | p05 | p25 | Median | p75 | p95 | Max |
|---|---|---|---|---|---|---|---|
| `source2` | 0.0000 | 0.3766 | 0.4618 | **0.8415** | 0.9412 | 0.9746 | 0.9969 |
| `source3` | 0.0000 | 0.3793 | 0.6833 | **0.8457** | 0.9394 | 0.9744 | 0.9962 |

**Key Finding**:
- The median Jaro-Winkler similarity is **~0.84**, meaning a massive fraction of missed pairs have substantial character-level overlap, but differ slightly in token ordering, prefixes, or spelling variations that prevent exact prefix/token blocking.

---

## 10. Address Variation Analysis

- **Exact normalized address equality**: Only 0.00% of missed pairs share identical address strings (all exact address matches were already captured by Rule B).
- **Address length difference**: Median character length difference is 14 characters, reflecting variable formatting (e.g. inclusion of suite numbers, floor numbers, or district names).

---

## 11. Script Analysis

Cataloged in `P1/reports/V2_MISSED_PAIR_SCRIPT_ANALYSIS.tsv`:

| Target Source | Total Missed | ASCII-Only | Pct ASCII | Devanagari Script | Pct Devanagari | Other Non-ASCII | Cross-Script Candidates | Pct Cross-Script |
|---|---|---|---|---|---|---|---|---|
| `source2` | 1,255,044 | 868,527 | 69.20% | 167,174 | 13.32% | 219,343 | 386,517 | **30.80%** |
| `source3` | 1,360,153 | 1,053,254 | 77.44% | 112,579 | 8.28% | 194,320 | 306,899 | **22.56%** |

**Key Finding**:
- **693,416 missed pairs (26.51%) feature cross-script representation**, where one entity uses Latin/ASCII characters and the counterpart uses Devanagari or other Indic script representations.
- Rule-based Latin prefix blocking is structurally incapable of bridging this gap without transliteration or phonetic indexing.

---

## 12. House Number Analysis

| Pattern | S2 Count | S2 Pct | S3 Count | S3 Pct | Total Pairs | Combined Pct |
|---|---|---|---|---|---|---|
| Exact house number match | 426,267 | 33.96% | 494,515 | 36.36% | 920,782 | 35.21% |
| Normalized house match (zero-stripped) | 451,155 | 35.95% | 515,938 | 37.93% | 967,093 | 36.98% |
| Differing house numbers | 480,045 | 38.25% | 497,318 | 36.56% | 977,363 | 37.37% |
| House number missing on one side | 277,456 | 22.11% | 278,755 | 20.49% | 556,211 | 21.27% |
| House number missing on both sides | 71,276 | 5.68% | 89,565 | 6.58% | 160,841 | 6.15% |

**Key Finding**:
- **717,052 missed pairs (27.42%) suffer from house number absence** on at least one side.
- For **967,093 pairs (36.98%)**, the normalized house number matches *perfectly*, but the pair was missed solely because the business name failed prefix4, token1, and root_token checks.

---

## 13. V2 Rule Failure Analysis

By evaluating each V2 rule predicate across the missed pairs:
- **Prefix4 match but house failed**: 553,416 S2 pairs (44.09%) and 601,895 S3 pairs (44.25%) share the 4-letter name prefix, but house numbers diverged or were missing.
- **House match but prefix4 failed**: 426,267 S2 pairs (33.96%) and 494,515 S3 pairs (36.36%) share house numbers, but business names vary past character 4.
- **House match but token1 failed**: 510,832 S2 pairs (40.70%) and 554,762 S3 pairs (40.79%) share house numbers, but first whitespace token differs.
- **Zip code match but prefix3 failed**: 537,114 S2 pairs (42.80%) and 588,289 S3 pairs (43.25%) share postal codes, but 3-character prefixes diverge.

---

## 14. Error Buckets

Cataloged in `P1/reports/V2_MISSED_PAIR_ERROR_BUCKETS.tsv`:

| Error Bucket | S2 Count | S2 Pct | S3 Count | S3 Pct | Total Volume | Pct of Missed | Root Mechanism |
|---|---|---|---|---|---|---|---|
| `high_name_similarity_house_divergence` | 294,881 | 23.50% | 324,122 | 23.83% | **619,003** | **23.67%** | Name $JW \ge 0.85$, but address house number differs or is unextracted |
| `missing_house_number_one_side` | 269,779 | 21.50% | 273,926 | 20.14% | **543,705** | **20.79%** | One entity lacks house digits, blocking Rules A, C, F, G |
| `house_match_moderate_name_variation` | 168,503 | 13.43% | 251,632 | 18.50% | **420,135** | **16.07%** | House matches, but name has token permutations ($0.60 \le JW < 0.85$) |
| `unstructured_divergence` | 164,861 | 13.14% | 185,993 | 13.67% | **350,854** | **13.42%** | Compound variation across both name and address tokens |
| `indic_devanagari_script` | 167,174 | 13.32% | 112,579 | 8.28% | **279,753** | **10.70%** | Cross-script / non-Latin text mismatch |
| `house_match_severe_name_variation` | 126,083 | 10.05% | 125,992 | 9.26% | **252,075** | **9.64%** | House matches, but name string severely differs ($JW < 0.60$) |
| `missing_house_number_both_sides` | 59,524 | 4.74% | 80,374 | 5.91% | **139,898** | **5.35%** | Rural/unnumbered commercial locations on both sides |
| `zip_match_prefix_mismatch` | 4,239 | 0.34% | 5,535 | 0.41% | **9,774** | **0.37%** | Postal code matches, but 3-char prefix fails |

---

## 15. Evidence-Based Findings

1. **Strict House Number Coupling is the Primary Blocker**: 44.5% of misses (1.16M pairs) are caused because blocking rules rigidly demand an exact house number match, yet addresses contain missing, descriptive, or divergent house digits.
2. **Name Variations are Often Minor Lexically**: Over 619,000 missed pairs have $JW \ge 0.85$ names; they were missed simply because candidate rules required an accompanying house number match.
3. **Indic / Devanagari Scripts Represent a Clear 10% Floor**: Nearly 280,000 missed pairs in India cannot be solved by standard Latin string heuristics.
4. **Country Partitioning is 100% Accurate**: Zero ground truth pairs cross country borders. All retrieval channels must maintain `country_s1 = country_target`.

---

## 16. Phase 3 Retrieval Experiments Recommended

Detailed in `P1/reports/PHASE3_RETRIEVAL_EXPERIMENT_PLAN.md`:
1. **`EXP-RET-01`**: Soft-Name Matching via Sparse Token / Character N-Gram Overlap (targeting 619k pairs in `high_name_similarity_house_divergence`).
2. **`EXP-RET-02`**: Relaxed Name Matching on Normalized House Key (targeting 420k pairs in `house_match_moderate_name_variation`).
3. **`EXP-RET-03`**: Cross-Script Indic Transliteration Key (targeting 280k pairs in `indic_devanagari_script`).
4. **`EXP-RET-04`**: Locality Shingle + Name Prefix Index for Unnumbered Entities (targeting 683k pairs with missing house numbers).

---

## 17. P1 → P2 Data Contract

The formal data interface contract is documented in `P1/reports/P1_P2_DATA_CONTRACT.md`. It defines exact Parquet schema requirements, deterministic ordering (`entity_id ASC`), zero-null guarantees, three-column candidate structures, and cryptographic verification protocols.

---

## 18. Artifacts Created

```
P1/data/entities/train/source1/train_s1_entities.parquet
P1/data/entities/train/source2/train_s2_entities.parquet
P1/data/entities/train/source3/train_s3_entities.parquet
P1/data/entities/test/source1/test_s1_entities.parquet
P1/data/entities/test/source2/test_s2_entities.parquet
P1/data/entities/test/source3/test_s3_entities.parquet
P1/experiments/phase1/v2_missed_pairs/missed_s2.tsv
P1/experiments/phase1/v2_missed_pairs/missed_s3.tsv
P1/manifests/entity_dataset_manifest.tsv
P1/reports/CANONICAL_ENTITY_SCHEMA.md
P1/reports/PHASE1_ENTITY_DATASET_VALIDATION.md
P1/reports/V2_MISSED_PAIR_COUNTRY_BREAKDOWN.tsv
P1/reports/V2_MISSED_PAIR_SCRIPT_ANALYSIS.tsv
P1/reports/V2_MISSED_PAIR_ERROR_BUCKETS.tsv
P1/reports/v2_missed_pairs_stats.json
P1/reports/PHASE3_RETRIEVAL_EXPERIMENT_PLAN.md
P1/reports/P1_P2_DATA_CONTRACT.md
P1/reports/PHASE1_REPORT.md
P1/scripts/data/build_entity_datasets.py
P1/scripts/data/validate_entity_datasets.py
P1/scripts/audit/analyze_v2_missed_pairs.py
```

---

## 19. Checksums

All Phase 1 artifacts are registered in `P1/checksums/SHA256SUMS.txt`:
- `train_s1_entities.parquet`: `22a84062c98605ff94b3bd8d29629be9319aca72da4d9f7a2be03692949dab31`
- `train_s2_entities.parquet`: `b8bcc95ddd650d78c7458ca320bbd1e1295d610d55f03f919a55f0d04516cc32`
- `train_s3_entities.parquet`: `f1e0ace13b59ba2a84e696b2bc6fce2e65ce8fca219f6ac9368525582f9fc6f8`
- `test_s1_entities.parquet`: `e6ff58d72b00001e9078094ec427799294cbe478a256c50eb3795ad1c8a5578c`
- `test_s2_entities.parquet`: `62c74f7dfc62adc0497b88e663a6f4979e5138d370b72f0db829619aa5a73efe`
- `test_s3_entities.parquet`: `f7748097ca62b213977de22ba3a29764a9c9697d9ea40857dbd7d06f9dcbaa8c`
- `missed_s2.tsv`: `3bcbef2a04d896ee7999d1c6ca628621d1a4b1a39cdef66650511133a55fc5f7`
- `missed_s3.tsv`: `8669f6d8c3a26fa5eac8f8a70906601b63977043e267fbc9890f2c5b6ab21f5b`
- `entity_dataset_manifest.tsv`: `68ecbde199c1de73f8a42aa07ed8a4b5208cd38fab5264aa93dceab4f69e85bf`

---

## 20. Known Limitations

1. **No Dense Embeddings Extracted**: Text representations are symbolic; semantic embeddings were intentionally deferred to maintain reproducibility and lightweight local execution.
2. **Transliteration Dependency**: Offline Indic transliteration libraries are not installed in the local Python environment; actual transliteration indexing will require dependency installation in Phase 3.
3. **No Candidate Generation in Phase 1**: Per design constraints, no candidate generation was executed; the candidate universe remains frozen at V2 pending Phase 2 validation.

---

## 21. Phase 1 Acceptance Checklist

- [x] Canonical train S1 dataset created (`train_s1_entities.parquet`)
- [x] Canonical train S2 dataset created (`train_s2_entities.parquet`)
- [x] Canonical train S3 dataset created (`train_s3_entities.parquet`)
- [x] Canonical test S1 dataset created (`test_s1_entities.parquet`)
- [x] Canonical test S2 dataset created (`test_s2_entities.parquet`)
- [x] Canonical test S3 dataset created (`test_s3_entities.parquet`)
- [x] Raw row counts preserved (100.0% retention across all 24.2M records)
- [x] Entity IDs preserved (exact 1-to-1 matching)
- [x] Duplicate IDs = 0
- [x] No entities silently dropped
- [x] Schema documented (`CANONICAL_ENTITY_SCHEMA.md`)
- [x] Determinism verified (ordered by `entity_id ASC`)
- [x] Entity manifest generated (`entity_dataset_manifest.tsv`)
- [x] SHA256 generated and recorded in `SHA256SUMS.txt`
- [x] V2 missed S2 pairs generated (`missed_s2.tsv`, 1,255,044 rows)
- [x] V2 missed S3 pairs generated (`missed_s3.tsv`, 1,360,153 rows)
- [x] Missed pairs verified against GT (100% in GT, 0% in V2)
- [x] Country analysis completed (`V2_MISSED_PAIR_COUNTRY_BREAKDOWN.tsv`)
- [x] Name analysis completed (Jaro-Winkler percentiles measured)
- [x] Address analysis completed (house number and address length analyzed)
- [x] Script analysis completed (`V2_MISSED_PAIR_SCRIPT_ANALYSIS.tsv`)
- [x] House-number analysis completed (37.0% exact house match; 27.4% missing)
- [x] V2 rule failure analysis completed (diagnosed across Rules A-I)
- [x] Error buckets documented (`V2_MISSED_PAIR_ERROR_BUCKETS.tsv`)
- [x] Phase 3 experiment plan created (`PHASE3_RETRIEVAL_EXPERIMENT_PLAN.md`)
- [x] P1/P2 data contract created (`P1_P2_DATA_CONTRACT.md`)
- [x] No V3 created
- [x] No production pipeline modified
- [x] No model modified
- [x] Phase 1 report created (`PHASE1_REPORT.md`)
- [x] Git commit created
