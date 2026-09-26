# Phase 1 — Canonical Entity Dataset Validation Report

**Date**: 2026-09-26  
**Validator**: P1 automated verification suite (`validate_entity_datasets.py`)  
**Standard**: Lossless row retention, exact 1-to-1 ID mapping, zero duplicates, deterministic sort order.  

---

## 1. Summary of Dataset Integrity

| Dataset | Split | Source | Raw Rows | Canonical Rows | Lost Rows | Duplicate IDs | Missing IDs | Extra IDs | Deterministic Order | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| `train_s1_entities` | train | source1 | 2,206,821 | 2,206,821 | 0 | 0 | 0 | 0 | YES (Ascending) | **PASS** |
| `train_s2_entities` | train | source2 | 5,034,616 | 5,034,616 | 0 | 0 | 0 | 0 | YES (Ascending) | **PASS** |
| `train_s3_entities` | train | source3 | 5,285,603 | 5,285,603 | 0 | 0 | 0 | 0 | YES (Ascending) | **PASS** |
| `test_s1_entities` | test | source1 | 1,732,544 | 1,732,544 | 0 | 0 | 0 | 0 | YES (Ascending) | **PASS** |
| `test_s2_entities` | test | source2 | 4,887,273 | 4,887,273 | 0 | 0 | 0 | 0 | YES (Ascending) | **PASS** |
| `test_s3_entities` | test | source3 | 5,082,316 | 5,082,316 | 0 | 0 | 0 | 0 | YES (Ascending) | **PASS** |

---

## 2. File Properties and Signatures

| Dataset | Path | Format | Size (MB) | Columns | SHA256 Signature |
|---|---|---|---|---|---|
| `train_s1_entities` | `P1/data/entities/train/source1/train_s1_entities.parquet` | Parquet (ZSTD) | 365.65 | 22 | `22a84062c98605ff94b3bd8d29629be9319aca72da4d9f7a2be03692949dab31` |
| `train_s2_entities` | `P1/data/entities/train/source2/train_s2_entities.parquet` | Parquet (ZSTD) | 863.84 | 22 | `b8bcc95ddd650d78c7458ca320bbd1e1295d610d55f03f919a55f0d04516cc32` |
| `train_s3_entities` | `P1/data/entities/train/source3/train_s3_entities.parquet` | Parquet (ZSTD) | 899.22 | 22 | `f1e0ace13b59ba2a84e696b2bc6fce2e65ce8fca219f6ac9368525582f9fc6f8` |
| `test_s1_entities` | `P1/data/entities/test/source1/test_s1_entities.parquet` | Parquet (ZSTD) | 292.95 | 22 | `e6ff58d72b00001e9078094ec427799294cbe478a256c50eb3795ad1c8a5578c` |
| `test_s2_entities` | `P1/data/entities/test/source2/test_s2_entities.parquet` | Parquet (ZSTD) | 862.86 | 22 | `62c74f7dfc62adc0497b88e663a6f4979e5138d370b72f0db829619aa5a73efe` |
| `test_s3_entities` | `P1/data/entities/test/source3/test_s3_entities.parquet` | Parquet (ZSTD) | 877.38 | 22 | `f7748097ca62b213977de22ba3a29764a9c9697d9ea40857dbd7d06f9dcbaa8c` |

---

## 3. Verification Conclusions

1. **Zero Row Loss**: All 24,228,873 entity records across train and test partitions were captured with 100.0% retention.
2. **Zero ID Corruption**: No ID mutations, null IDs, or truncated identifiers were observed.
3. **Zero ID Duplication**: All unique entity IDs match the source count exactly.
4. **Deterministic Storage**: Every file is ordered by `entity_id ASC`, ensuring repeatable row traversal for downstream candidate generation and feature extraction (note: deterministic row ordering provides stable traversal, but does not constitute proof of bitwise binary reproducibility across diverse writer libraries).
5. **Dual Representation**: Raw and normalized strings are preserved side-by-side, guaranteeing full provenance for future retrieval models.
6. **Legitimate NULL / Empty Derived Values**: Non-extractable fields (e.g. absent house numbers or missing postal codes) evaluate legitimately to empty strings/NULLs, while primary key `entity_id` is strictly non-null.
