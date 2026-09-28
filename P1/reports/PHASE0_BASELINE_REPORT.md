# Phase 0 — P1 Foundation & Baseline Report

**Date**: 2026-09-26  
**Author**: Person 1 (P1)  
**Scope**: Workspace initialization, canonical artifact verification, pipeline inventory, and baseline freeze  
**Phase Status**: COMPLETED  

---

## 1. Objective

The primary objective of Phase 0 is to establish a rigorous, reproducible P1 workspace and freeze the verified baseline prior to candidate-generation and modeling improvements. 

Under the Phase 0 operational contract:
- No pipeline optimizations or model retraining are performed.
- No V3 candidate generation or new retrieval techniques (TF-IDF, dense embeddings, phonetic matching) are introduced.
- Existing production logic remains untouched.
- All numbers, paths, schemas, and metrics are derived strictly from actual repository contents.

---

## 2. Repository State

The repository contains three primary project modules and supporting directories:
- `P1/`: Person 1 workspace (data integrity, normalization, candidate universe).
- `P2/`: Person 2 workspace (candidate generation V1/V2, feature extraction, LightGBM modeling).
- `P3/`: Person 3 validation framework (frozen 5-fold partition manifest, official competition scorer).
- `data/`: Raw source TSVs and split `.part_*` files.
- `outputs/person1_step1/`: Initial Person 1 exploratory artifacts, normalized datasets, and baseline Strategy B test candidates.
- `validation/`: Canonical competition scorer (`validation/scorer_v1.py`) and metric logic (`validation/metrics.py`).
- `audit/`: Independent forensic audit report (`audit/FULL_PIPELINE_AUDIT.md`) and core verification scripts.

---

## 3. Git State

- **Current Branch**: `main` (synchronized with `origin/main`)
- **Baseline Commit SHA**: `49a2f1025a17ca164eebfa136d8590c9b0e27c19`
- **Commit Message**: `feat: add cands_BCD_v2 candidate generation script and comprehensive analysis reports`
- **Recent Git Log**:
  - `49a2f10`: feat: add cands_BCD_v2 candidate generation script and comprehensive analysis reports
  - `5cfbaa1`: Merge pull request #1 from Ayushxsingh100/ayush
  - `64ae3bc`: feat(P2): expose full canonical E02 OOF predictions via 5 fold-specific LFS shards
  - `d202088`: Record E02 baseline and full OOF metadata
  - `308d5ff`: Update gitignore
- **Safety Status**: No commits were reset, rebased, or forced. Existing branches and historical records remain intact.

---

## 4. P1 Directory Structure

The clean P1 workspace structure has been created without unnecessary data duplication:

```
P1/
├── README.md                              # P1 charter, baseline numbers, canonical paths
├── data/
│   ├── README.md                          # Data referencing policy (no >10GB redundancy)
│   ├── raw/{train,test}/                  # References to canonical data/
│   ├── normalized/{train,test}/           # References to outputs/person1_step1/normalized/
│   ├── entities/{train,test}/             # Entity partitions (Source1, Source2, Source3)
│   ├── ground_truth/                      # Reference to data/train/train_ground_truth.tsv
│   └── candidates/
│       ├── v1/                            # References to P2/data/candidates/ (V1)
│       ├── v2/                            # References to P2/data/candidates/ (V2)
│       └── v3/
│           └── README.md                  # Explicit placeholder: V3 not implemented in Phase 0
├── scripts/
│   ├── audit/                             # Baseline verification and artifact generator
│   │   ├── verify_baseline.py
│   │   └── generate_phase0_artifacts.py
│   ├── data/                              # Data transformation utilities (Phase 1)
│   ├── candidates/                        # Candidate generation scripts (Phase 2/3)
│   └── utils/                             # Common helper utilities
├── configs/
│   └── phase0_baseline.yaml               # Canonical paths and baseline config
├── manifests/
│   ├── data_manifest.tsv                  # Exhaustive manifest of all raw/normalized datasets
│   ├── artifact_manifest.tsv              # Inventory of all authoritative artifacts
│   ├── script_registry.tsv                # Registry of all repo scripts and usage status
│   └── REPRODUCIBILITY.md                 # System hardware, Python, DuckDB, package specs
├── reports/
│   ├── PHASE0_BASELINE_REPORT.md          # This report
│   ├── DATASET_INTEGRITY_BASELINE.tsv     # Exact row counts, unique IDs, nulls
│   ├── GROUND_TRUTH_BASELINE.tsv          # Exact GT metrics and exploded pair counts
│   ├── NORMALIZATION_BASELINE.tsv         # Normalization retention and cleaning metrics
│   ├── CANDIDATE_ARTIFACT_BASELINE.tsv    # V1, V2, and test candidate statistics
│   ├── PIPELINE_COMPONENT_STATUS.tsv      # Classification of active vs stale components
│   └── verified_baseline_cache.json       # Raw JSON cache of all direct measurements
├── experiments/
│   ├── phase0/                            # Phase 0 experiment documentation
│   ├── phase1/                            # Reserved for Phase 1
│   ├── phase2/                            # Reserved for Phase 2
│   └── phase3/                            # Reserved for Phase 3
├── logs/                                  # Execution logs
└── checksums/
    └── SHA256SUMS.txt                     # Cryptographic SHA256 hashes of all key artifacts
```

---

## 5. Canonical Datasets

All raw datasets were reconstructed from Git LFS split chunks using `reconstruct_data.sh`. Row counts, unique ID counts, and schemas have been verified independently via DuckDB.

| Dataset | Split | Source | Rows | Unique IDs | Duplicate IDs | Null / Empty IDs | File Size (Bytes) | SHA256 Hash | Status |
|---|---|---|---|---|---|---|---|---|---|
| `train_source1` | Train | S1 | 2,206,821 | 2,206,821 | 0 | 0 | 210,069,713 | `591af0e1dfeb65cab71ea6ee8cb69df00f92d6ba6fa79e05746c938775d14973` | VERIFIED_NOW |
| `train_source2` | Train | S2 | 5,034,616 | 5,034,616 | 0 | 0 | 489,301,488 | `6336c1a055eec79cf8a6d99fdc8d32a2e4d9dc2662e00963cb35d66b89ed09ed` | VERIFIED_NOW |
| `train_source3` | Train | S3 | 5,285,603 | 5,285,603 | 0 | 0 | 503,705,637 | `67da22f5151898ff3006febd836c1a159e97ae95efa7257a5aff4fda685e58e9` | VERIFIED_NOW |
| `test_source1` | Test | S1 | 1,732,544 | 1,732,544 | 0 | 0 | 175,022,086 | `3d4a32c54c2ca9c53fd7c2be105bf26f708f94c4d2f88eb370972a195665c2f5` | VERIFIED_NOW |
| `test_source2` | Test | S2 | 4,887,273 | 4,887,273 | 0 | 0 | 509,456,422 | `79d906c7497af2ace70aa277f6e334a652094909de99bd6c57b53420b6a7b2dd` | VERIFIED_NOW |
| `test_source3` | Test | S3 | 5,082,316 | 5,082,316 | 0 | 0 | 506,002,772 | `850942b11d2a4343486ed0834e28bce9f3b385f3fd497fd60ccf4ea3b8bda035` | VERIFIED_NOW |
| `train_ground_truth` | Train | GT | 2,206,821 | 2,206,821 | 0 | 0 | 127,015,583 | `70bc1d8a16c667e0155c2105d0ab2ebe41d7e7a85d8a529e3ca81c6c3a5af037` | VERIFIED_NOW |

**Schema Consistency**:
- All Source 1, 2, and 3 datasets have identical columns: `entity_id`, `business_name`, `business_address`, `country`.
- Ground truth has columns: `source1_entity_id`, `matched_entity_ids`.

---

## 6. Ground Truth

Ground truth integrity was verified on `data/train/train_ground_truth.tsv` and compared against `outputs/person1_step1/train_ground_truth_reconstructed.tsv`.

| Metric | Measured Value | Historical Value | Match | Status |
|---|---|---|---|---|
| Total Ground Truth Rows | 2,206,821 | 2,206,821 | YES | VERIFIED_NOW |
| Unique Source 1 IDs in GT | 2,206,821 | 2,206,821 | YES | VERIFIED_NOW |
| Duplicate S1 IDs | 0 | 0 | YES | VERIFIED_NOW |
| S1 with Empty / No Match | 123,247 (5.5848%) | 123,247 | YES | VERIFIED_NOW |
| Total Exploded Match Pairs | 7,638,365 | 7,638,365 | YES | VERIFIED_NOW |
| S2 True Match Pairs | 3,693,619 (48.36%) | 3,693,619 | YES | VERIFIED_NOW |
| S3 True Match Pairs | 3,944,746 (51.64%) | 3,944,746 | YES | VERIFIED_NOW |
| Non-S2 / Non-S3 Target Pairs | 0 (0.00%) | 0 | YES | VERIFIED_NOW |
| S1 Matched to Both S2 & S3 | 1,776,047 (80.48%) | 1,776,047 | YES | VERIFIED_NOW |
| S1 Matched to S2 Only | 143,029 (6.48%) | 143,029 | YES | VERIFIED_NOW |
| S1 Matched to S3 Only | 164,498 (7.45%) | 164,498 | YES | VERIFIED_NOW |
| Raw GT SHA256 = Reconstructed SHA256 | `70bc1d8a...` | `70bc1d8a...` | YES | VERIFIED_NOW |

**Verdict**: PASS. Ground truth is intact, clean, and structurally consistent.

---

## 7. Normalization

Normalized datasets located in `outputs/person1_step1/normalized/` apply in-place string cleaning (case folding, legal entity standardization, whitespace normalization).

| Normalized Dataset | Raw Input Rows | Normalized Rows | Row Retention | Missing Names | Missing Addresses | Missing Country | Status |
|---|---|---|---|---|---|---|---|
| `train_source1_normalized` | 2,206,821 | 2,206,821 | 100.0% (0 lost) | 0 | 0 | 0 | VERIFIED_NOW |
| `train_source2_normalized` | 5,034,616 | 5,034,616 | 100.0% (0 lost) | 6 | 168,967 | 0 | VERIFIED_NOW |
| `train_source3_normalized` | 5,285,603 | 5,285,603 | 100.0% (0 lost) | 18 | 175,916 | 0 | VERIFIED_NOW |
| `test_source1_normalized` | 1,732,544 | 1,732,544 | 100.0% (0 lost) | 0 | 0 | 0 | VERIFIED_NOW |
| `test_source2_normalized` | 4,887,273 | 4,887,273 | 100.0% (0 lost) | 49 | 129,408 | 0 | VERIFIED_NOW |
| `test_source3_normalized` | 5,082,316 | 5,082,316 | 100.0% (0 lost) | 61 | 136,098 | 0 | VERIFIED_NOW |

**Row Retention**: Exact 100% preservation across all 6 files. No rows or entity IDs were dropped.  
**Note**: Normalization was performed in-place (same column headers: `entity_id`, `business_name`, `business_address`, `country`).

---

## 8. Candidate Artifacts

Existing candidate pair files across V1, V2, and legacy Person 1 outputs were verified directly.

| Candidate Artifact | Split | Target | Total Pairs | Unique S1 | Unique Target | Positives | Negatives | Pair Recall | Status |
|---|---|---|---|---|---|---|---|---|---|
| `train_candidate_pairs_s2.tsv` | Train | S2 | 24,594,064 | 1,859,868 | 2,798,463 | 2,214,137 | 22,379,927 | **59.9449%** | VERIFIED_NOW |
| `train_candidate_pairs_s3.tsv` | Train | S3 | 29,998,661 | 1,881,377 | 3,016,380 | 2,346,442 | 27,652,219 | **59.4827%** | VERIFIED_NOW |
| **V1 Train Total** | **Train** | **Both** | **54,592,725** | — | — | **4,560,579** | **50,032,146** | **59.7062%** | VERIFIED_NOW |
| `train_candidate_pairs_s2_v2.tsv`| Train | S2 | 30,359,040 | 1,929,373 | 3,045,029 | 2,438,575 | 27,920,465 | **66.0213%** | VERIFIED_NOW |
| `train_candidate_pairs_s3_v2.tsv`| Train | S3 | 36,973,484 | 1,944,749 | 3,277,458 | 2,584,593 | 34,388,891 | **65.5199%** | VERIFIED_NOW |
| **V2 Train Total** | **Train** | **Both** | **67,332,524** | — | — | **5,023,168** | **62,309,356** | **65.7623%** | VERIFIED_NOW |
| `test_candidate_pairs_s2.tsv` (V1) | Test | S2 | 30,232,352 | 1,478,061 | 2,506,067 | — | — | — | VERIFIED_NOW |
| `test_candidate_pairs_s3.tsv` (V1) | Test | S3 | 35,822,910 | 1,493,071 | 2,725,472 | — | — | — | VERIFIED_NOW |
| `test_candidate_pairs_s2_v2.tsv` | Test | S2 | 34,919,169 | 1,526,337 | 2,712,234 | — | — | — | VERIFIED_NOW |
| `test_candidate_pairs_s3_v2.tsv` | Test | S3 | 41,714,627 | 1,537,152 | 2,946,450 | — | — | — | VERIFIED_NOW |
| `outputs/person1_step1/test_candidate_pairs_s2.tsv` | Test | S2 | 26,046,195 | 1,331,914 | 2,298,136 | — | — | — | VERIFIED_NOW |
| `outputs/person1_step1/test_candidate_pairs_s3.tsv` | Test | S3 | 30,777,878 | 1,350,577 | 2,492,891 | — | — | — | VERIFIED_NOW |

---

## 9. Active Pipeline

The authoritative, end-to-end active pipeline is established as follows:

```
[Raw Parts] -> reconstruct_data.sh -> [Raw TSVs: data/{train,test}/*.tsv]
     │
     ▼
[outputs/person1_step1/normalized/*.tsv]
     │
     ▼
[P2/scripts/cands_BCD_v1_tasks.py] (Rules A, B, C, D via DuckDB SQL)
     │
     ├──> V1 Train Candidates (54,592,725 pairs in P2/data/candidates/)
     └──> V1 Test Candidates  (66,055,262 pairs in P2/data/candidates/)
     │
     ▼
[P2/scripts/e02_train_lgb.py] (LightGBM 5-fold CV, 10% negative downsampling, seed 2026)
     │
     ├──> 5 Fold Models (ran on training node)
     └──> Full Canonical OOF Predictions (P2/reports/E02_OOF_FULL_CANONICAL_fold{0..4}.tsv)
     │
     ▼
[validation/scorer_v1.py] (Evaluates joint S1 predictions against train_ground_truth.tsv)
     │
     └──> Macro F0.5 = 0.769452 (T=0.5)
```

---

## 10. Stale / Obsolete Pipeline Components

| Component | Path | Status | Finding / Reason |
|---|---|---|---|
| `finish_pipeline.py` | `P2/scripts/finish_pipeline.py` | OBSOLETE | Uses house number regex `\b[0-9]+\b` (inconsistent with `[0-9]+[A-Za-z]?` in canonical pipeline) and references stale Strategy B candidates. |
| `phase3_to_8_pipeline.py` | `P2/scripts/phase3_to_8_pipeline.py` | OBSOLETE | Hardcodes non-existent paths (`train_strategy_b_candidates_*.tsv`). Superseded by E02 scripts. |
| Person 1 Test Candidates | `outputs/person1_step1/test_candidate_pairs_s*.tsv` | STALE | Only contains Strategy B candidates (Rules A+B; 26.0M S2 / 30.8M S3). Inconsistent with V1 train candidates. |
| Python blocking utility | `src/blocking.py` | STALE | Unused by production pipeline; pipeline executes blocking logic directly in DuckDB SQL. |
| Python normalization utility | `src/normalization.py` | STALE | Unused by production pipeline. |
| Python feature utility | `src/features.py` | STALE | Unused by production pipeline. |

---

## 11. Checksums

Authoritative SHA256 checksums are documented in `P1/checksums/SHA256SUMS.txt`. Key signatures include:
- `data/train/train_ground_truth.tsv`: `70bc1d8a16c667e0155c2105d0ab2ebe41d7e7a85d8a529e3ca81c6c3a5af037`
- `outputs/person1_step1/train_ground_truth_reconstructed.tsv`: `70bc1d8a16c667e0155c2105d0ab2ebe41d7e7a85d8a529e3ca81c6c3a5af037`
- `P3/reports/folds_v1_manifest.tsv`: `9dcec5d83a477d224067e71b93abc21af8befa26f9c399568121fa83ba8801a3`
- `P2/data/candidates/train_candidate_pairs_s2.tsv`: `1a1f671d500bea88095cb69f5d32dabe6644357ff91432526305b10856f210be`
- `P2/data/candidates/train_candidate_pairs_s3.tsv`: `b34f05adaefd6990495ebf2581f9d1b0582a15bb71a87d05a54c9caf6c440255`
- `P2/data/candidates/train_candidate_pairs_s2_v2.tsv`: `386c0a0c0dacda2df90d26778a893c760b686c1c4e84bc0ef033e7ce84c28d22`
- `P2/data/candidates/train_candidate_pairs_s3_v2.tsv`: `6d6cc5171beb4f445a4016f35a27e488e543cdeaa66bb5a26b342f87302c8404`

---

## 12. Reproducibility

- **Compute Engines**:
  - DuckDB 1.5.5 handles candidate generation, SQL feature engineering, and high-volume joins without memory overflow.
  - Python 3.13.5 provides orchestration and metric evaluation.
- **Hardware Profile**: Apple M4 (16 GB unified memory), running Darwin 27.0.0 (macOS arm64).
- **Modeling Framework Dependency**: `lightgbm` is currently not installed on this local macOS environment (`pandas 3.0.5`, `numpy 2.5.2`, `pyarrow 25.0.1`, `scikit-learn 1.9.0` are present). LightGBM was executed on an AMD64 Windows training node. Installing `lightgbm` locally will be necessary when local retraining is initiated.

---

## 13. Verified Baseline Metrics

| Metric | Value | Verification Status | Source / Evidence |
|---|---|---|---|
| **E02 OOF Macro $F_{0.5}$ (T=0.5)** | **0.769452** | VERIFIED_NOW | Verified from `P2/reports/E02_THRESHOLD_SWEEP.tsv` and 5 canonical OOF prediction files (`P2/reports/E02_OOF_FULL_CANONICAL_fold*.tsv`) |
| **V1 Candidate Oracle $F_{0.5}$** | **0.784523** | VERIFIED_NOW | Verified from `P2/reports/E02_RUN.json` and candidate labels |
| **Model Efficiency Gap** | **0.015071** | VERIFIED_NOW | $0.784523 - 0.769452 = 0.015071$ |
| **Candidate Recall Ceiling Loss** | **0.215477** | VERIFIED_NOW | $1.000000 - 0.784523 = 0.215477$ |
| **V1 S2 True Pair Recall** | **59.9449%** | VERIFIED_NOW | 2,214,137 captured / 3,693,619 true pairs |
| **V1 S3 True Pair Recall** | **59.4827%** | VERIFIED_NOW | 2,346,442 captured / 3,944,746 true pairs |
| **V2 S2 True Pair Recall** | **66.0213%** | VERIFIED_NOW | 2,438,575 captured / 3,693,619 true pairs |
| **V2 S3 True Pair Recall** | **65.5199%** | VERIFIED_NOW | 2,584,593 captured / 3,944,746 true pairs |
| **Structural No-Match Floor** | **5.5848%** | VERIFIED_NOW | 123,247 S1 entities have zero true matches in GT |
| **Historical Test Public Score** | *N/A (Not submitted)* | HISTORICAL / NOT_VERIFIED | No leaderboard submission recorded in repo |

---

## 14. Known Issues

1. **Candidate Recall is the Dominant Bottleneck**:
   - V1 pair recall is ~59.7%, capping model performance at 0.7845.
   - V2 expands recall to ~65.8% (+462,589 positives), but remains unused in E02 modeling.
2. **Train/Test Candidate Generation Disconnect**:
   - `finish_pipeline.py` and `outputs/person1_step1/` use Strategy B candidates (Rules A+B only; 26.0M/30.8M pairs), whereas train candidates use Rules A, B, C, D (54.6M pairs).
3. **Regex Inconsistency in Legacy Scripts**:
   - `finish_pipeline.py` extracts house numbers via `\b[0-9]+\b` while canonical candidate generators use `[0-9]+[A-Za-z]?`.
4. **Missing Local Modeling Binaries**:
   - Trained LightGBM model files (`.lgb` / `.pkl`) are absent from local disk (generated on external Windows machine).

---

## 15. Missing Artifacts

| Artifact | Type | Expected Location | Impact | Status |
|---|---|---|---|---|
| Trained LightGBM Fold Models | Model binaries | `P2/models/*.txt` or `*.lgb` | Cannot infer directly without retraining or transferring model files | MISSING |
| Intermediate Training Feature Matrices | Precomputed features | `P2/data/features/*.parquet` | Cannot skip feature extraction during local retraining | MISSING |
| Final Test Submission File | Prediction submission | `outputs/matching_results.tsv` | No submission file ready for upload | MISSING |

---

## 16. Phase 0 Acceptance Checklist

- [x] **Git state recorded**: Clean working tree, commit `49a2f10` on `main`.
- [x] **P1 directory created**: Full standard hierarchy with no redundant massive data copies.
- [x] **Raw datasets identified**: `data/train/` and `data/test/` verified.
- [x] **Normalized datasets identified**: `outputs/person1_step1/normalized/` verified.
- [x] **Ground truth identified**: `data/train/train_ground_truth.tsv` verified.
- [x] **V1 candidates identified**: `P2/data/candidates/train_candidate_pairs_s{2,3}.tsv` verified.
- [x] **V2 candidates identified**: `P2/data/candidates/train_candidate_pairs_s{2,3}_v2.tsv` verified.
- [x] **Dataset row counts verified**: Exact counts for all 6 source datasets verified via DuckDB.
- [x] **ID integrity verified**: 0 duplicate IDs, 0 null IDs across all datasets.
- [x] **GT integrity verified**: 2,206,821 rows, 7,638,365 pairs, exact SHA256 match.
- [x] **SHA256 recorded where possible**: Cataloged in `P1/checksums/SHA256SUMS.txt`.
- [x] **Active pipeline identified**: End-to-end flow from reconstruction through scoring documented.
- [x] **Stale pipeline identified**: Classified in `P1/reports/PIPELINE_COMPONENT_STATUS.tsv`.
- [x] **Reproducibility environment recorded**: Captured in `P1/manifests/REPRODUCIBILITY.md`.
- [x] **No production logic modified**: All existing scripts in `P2/`, `P3/`, `validation/` preserved intact.
- [x] **No V3 generated**: Placeholder README created under `P1/data/candidates/v3/`.
- [x] **No model changes**: No retraining or hyperparameter modifications attempted.
- [x] **No threshold changes**: Preserved official $T=0.5$ baseline.
- [x] **Baseline report created**: Complete 17-section documentation in `P1/reports/PHASE0_BASELINE_REPORT.md`.

---

## 17. Phase 1 Starting Point

With Phase 0 complete and the baseline frozen, Phase 1 will focus on:
1. **Canonical Data Foundation**:
   - Establish formal P1 data loader and entity dataset preparation under `P1/scripts/data/`.
   - Verify non-lossy character encoding, script detection (Latin vs Indic), and token normalization across all 3 sources.
2. **Analysis of Uncaptured Ground Truth**:
   - Systematically inspect the 2.61M missed pairs remaining after V2 candidate generation.
   - Profile the exact failure modes (severe name spelling variations, missing address tokens, script differences).
3. **Formal Handoff Interface**:
   - Establish the standard entity schema specification and candidate format contract for Person 2.
