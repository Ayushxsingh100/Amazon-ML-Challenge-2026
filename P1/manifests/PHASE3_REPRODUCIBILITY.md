# Phase 3 Reproducibility Manifest & Execution Guide

**Project**: Amazon ML Challenge 2026 — Entity Resolution
**Phase**: Phase 3 Candidate Retrieval Improvement & Production V3 Promotion
**Date**: September 27, 2026
**Status**: **REPRODUCIBLE & VALIDATED**

---

## 1. System & Environment Specifications

- **OS**: macOS Darwin Kernel Version 24.6.0 (ARM64)
- **Python**: 3.13.5
- **DuckDB**: 1.5.5
- **Hardware Resources Used**: 8 execution threads, up to 10 GB DuckDB memory allocation

---

## 2. Canonical Input Artifacts

| Dataset Name | Split | Path | Format | Rows |
|:---|:---|:---|:---|---:|
| Source 1 Entities | Train | `P1/data/entities/train/source1/train_s1_entities.parquet` | Parquet | 2,206,821 |
| Source 2 Entities | Train | `P1/data/entities/train/source2/train_s2_entities.parquet` | Parquet | 5,034,616 |
| Source 3 Entities | Train | `P1/data/entities/train/source3/train_s3_entities.parquet` | Parquet | 5,285,603 |
| Source 1 Entities | Test | `P1/data/entities/test/source1/test_s1_entities.parquet` | Parquet | 1,732,544 |
| Source 2 Entities | Test | `P1/data/entities/test/source2/test_s2_entities.parquet` | Parquet | 4,887,273 |
| Source 3 Entities | Test | `P1/data/entities/test/source3/test_s3_entities.parquet` | Parquet | 5,082,316 |
| Ground Truth (Frozen) | Train | `data/train/train_ground_truth.tsv` | TSV | 7,638,365 pairs |
| V2 Candidates S2 | Train | `P2/data/candidates/train_candidate_pairs_s2_v2.tsv` | TSV | 35,979,481 |
| V2 Candidates S3 | Train | `P2/data/candidates/train_candidate_pairs_s3_v2.tsv` | TSV | 40,654,315 |
| V2 Candidates S2 | Test | `P2/data/candidates/test_candidate_pairs_s2_v2.tsv` | TSV | 34,919,169 |
| V2 Candidates S3 | Test | `P2/data/candidates/test_candidate_pairs_s3_v2.tsv` | TSV | 41,714,627 |

---

## 3. End-to-End Execution Sequence

### Step 1: Component Experiments Execution
To reproduce standalone candidate generation for experiments EXP-RET-01 through 04:
```bash
python3 P1/experiments/phase3/EXP-RET-01/run_exp_ret_01.py
python3 P1/experiments/phase3/EXP-RET-02/run_exp_ret_02.py
python3 P1/experiments/phase3/EXP-RET-03/run_exp_ret_03.py
python3 P1/experiments/phase3/EXP-RET-04/run_exp_ret_04.py
```

### Step 2: Component-Level Fanout Attribution Analysis
To attribute TEST combined fanout across configurations A through G:
```bash
python3 P1/scripts/retrieval/attribute_test_fanout.py
```
Output: `P1/reports/test_fanout_attribution.json`

### Step 3: Controlled Cap Sweep (Selection of Experiment D)
To execute the multi-cap sweep across train recall and test fanout:
```bash
python3 P1/experiments/phase3/EXP-RET-02_03_CAP_SWEEP/run_cap_sweep.py
```
Outputs:
- `P1/experiments/phase3/EXP-RET-02_03_CAP_SWEEP/experiment_summary.tsv`
- `P1/experiments/phase3/EXP-RET-02_03_CAP_SWEEP/metrics.json`

### Step 4: Authoritative Production V3 Generation
To generate the final capped candidate files for both train and test:
```bash
python3 P1/scripts/retrieval/generate_v3_candidates.py --split all
```
Outputs:
- `P1/data/candidates/v3/train_candidate_pairs_s2_v3.tsv`
- `P1/data/candidates/v3/train_candidate_pairs_s3_v3.tsv`
- `P1/data/candidates/v3/test_candidate_pairs_s2_v3.tsv`
- `P1/data/candidates/v3/test_candidate_pairs_s3_v3.tsv`

### Step 5: Canonical Evaluation & Integrity Audit
To independently evaluate the generated production V3 files:
```bash
python3 P1/scripts/evaluation/evaluate_candidate_set.py --candidate-file P1/data/candidates/v3/train_candidate_pairs_s2_v3.tsv --split train --output-report P1/reports/v3_train_s2_eval.json
python3 P1/scripts/evaluation/evaluate_candidate_set.py --candidate-file P1/data/candidates/v3/train_candidate_pairs_s3_v3.tsv --split train --output-report P1/reports/v3_train_s3_eval.json
python3 P1/scripts/evaluation/evaluate_candidate_set.py --candidate-file P1/data/candidates/v3/test_candidate_pairs_s2_v3.tsv --split test --output-report P1/reports/v3_test_s2_eval.json
python3 P1/scripts/evaluation/evaluate_candidate_set.py --candidate-file P1/data/candidates/v3/test_candidate_pairs_s3_v3.tsv --split test --output-report P1/reports/v3_test_s3_eval.json
```

---

## 4. Production V3 Candidate Artifact Manifest

| File Path | Split | Target | Rows | SHA256 Checksum |
|:---|:---|:---|---:|:---|
| `P1/data/candidates/v3/train_candidate_pairs_s2_v3.tsv` | Train | S2 | 42,933,945 | `61d5dce6389bf424daf61f842ed0f41ebc3991fa27763d4641c4ab4931a7508c` |
| `P1/data/candidates/v3/train_candidate_pairs_s3_v3.tsv` | Train | S3 | 50,238,004 | `d5e90bb6d9a5d81154a83366d98211352e95bebcbc84c5011f14cd0bfe10eeeb` |
| `P1/data/candidates/v3/test_candidate_pairs_s2_v3.tsv` | Test | S2 | 43,841,928 | `4aa0c71ea70a98aff77088da2c4b070e8fb305b8fbebfbda4daa4fa14398dfa2` |
| `P1/data/candidates/v3/test_candidate_pairs_s3_v3.tsv` | Test | S3 | 51,354,667 | `f24041fbbc3a5f56a4b9c26b20dfde3377770f85427a9373bc9c1efc8bf441cd` |

---

## 5. Verification Checksums

All checksums are verified with `shasum -a 256 <file>`. Candidate pair counts and content are byte-for-byte deterministic across runs.
