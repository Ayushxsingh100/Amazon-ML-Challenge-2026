# Reproducibility Metadata — Phase 0

## Environment Specifications (Local Verification Environment)

| Parameter | Value |
|---|---|
| **Operating System** | macOS Darwin 27.0.0 (Darwin Kernel Version 27.0.0, arm64) |
| **Hardware / CPU** | Apple M4 |
| **System Memory (RAM)** | 16 GB (17,179,869,184 bytes) |
| **Python Version** | Python 3.13.5 |
| **DuckDB Version** | 1.5.5 |
| **Git Current Branch** | `main` |
| **Git Baseline Commit SHA** | `49a2f1025a17ca164eebfa136d8590c9b0e27c19` |
| **Execution Date** | 2026-09-26 |

## Key Package Versions (Local)

| Package | Version | Status |
|---|---|---|
| `duckdb` | 1.5.5 | Installed (Primary compute engine) |
| `pandas` | 3.0.5 | Installed |
| `numpy` | 2.5.2 | Installed |
| `pyarrow` | 25.0.1 | Installed |
| `scipy` | 1.18.1 | Installed |
| `scikit-learn` | 1.9.0 | Installed |
| `lightgbm` | *Not installed locally* | **Missing locally** (used for E02 training on Windows) |
| `polars` | *Not installed locally* | N/A |

## Historical Training Node (from `P2/reports/E02_RUN.json`)

| Parameter | Historical Node Value |
|---|---|
| **OS** | Windows 10/11 (AMD64) |
| **Python Version** | 3.14.3 (tags/v3.14.3:323c59a) |
| **LightGBM Version** | 4.7.0 |
| **Training Execution Date** | 2026-09-26 |
| **Runtime** | 1994.29 seconds (~33.2 minutes) |
| **Negative Downsampling** | 10% deterministic sampling |

## Canonical Data Locality & Large File Strategy
- **Raw Parts**: Tracked via Git LFS in `data/train/` and `data/test/`.
- **Reconstruction**: Deterministic assembly via `reconstruct_data.sh`.
- **Large Datasets**: Rather than duplicating >10GB of data into `P1/data/`, canonical absolute paths are registered in `P1/manifests/data_manifest.tsv` and `P1/configs/phase0_baseline.yaml`.
- **Candidate Files**: Stored centrally in `P2/data/candidates/` and registered with exact SHA256 hashes in `P1/checksums/SHA256SUMS.txt`.
- **Full OOF Prediction Shards**: 5 fold files stored in `P2/reports/E02_OOF_FULL_CANONICAL_fold{0..4}.tsv` (total 54,592,725 rows).
