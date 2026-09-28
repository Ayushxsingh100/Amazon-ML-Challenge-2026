# Phase 2 Reproducibility Manifest

**Author**: Person 1 (P1)  
**Date**: September 26, 2026  
**Repository Commit**: `91bea2e`  
**Execution Environment**:
- Operating System: macOS Darwin 24.5.0 (arm64, Apple Silicon M4)
- Python Version: Python 3.13.5
- DuckDB Version: DuckDB 1.5.5
- Git LFS: Active (all `*.tsv` and `*.parquet` files tracked via Git LFS)

---

## 1. Reproduction Commands

### A. Run Candidate Set Evaluator (Individual Files)
```bash
# Evaluate Train V1 S2 Candidates
python3 P1/scripts/evaluation/evaluate_candidate_set.py \
  --candidate-file P2/data/candidates/train_candidate_pairs_s2.tsv \
  --candidate-source S2 \
  --split train

# Evaluate Train V2 S2 Candidates
python3 P1/scripts/evaluation/evaluate_candidate_set.py \
  --candidate-file P2/data/candidates/train_candidate_pairs_s2_v2.tsv \
  --candidate-source S2 \
  --split train

# Evaluate Test V2 S2 Candidates
python3 P1/scripts/evaluation/evaluate_candidate_set.py \
  --candidate-file P2/data/candidates/test_candidate_pairs_s2_v2.tsv \
  --candidate-source S2 \
  --split test
```

### B. Generate Complete Phase 2 Baseline & Schema Reports
```bash
python3 P1/scripts/evaluation/generate_phase2_reports.py
```
This runs the full multi-file audit across all 10 candidate files, computes ground truth recall, fanout quantiles, and schema integrity, and writes:
- `P1/reports/PHASE2_CANDIDATE_BASELINE.tsv`
- `P1/reports/PHASE2_CANDIDATE_SCHEMA_VALIDATION.tsv`

---

## 2. Input Datasets & Checksums

| Dataset Path | Rows | Size (bytes) | SHA256 Checksum |
|---|---|---|---|
| `data/train/train_ground_truth.tsv` | 2,206,821 | 127,015,583 | `70bc1d8a16c667e0155c2105d0ab2ebe41d7e7a85d8a529e3ca81c6c3a5af037` |
| `P1/data/entities/train/source1/train_s1_entities.parquet` | 2,206,821 | 383,413,239 | `22a84062c98605ff94b3bd8d29629be9319aca72da4d9f7a2be03692949dab31` |
| `P1/data/entities/train/source2/train_s2_entities.parquet` | 5,034,616 | 905,805,958 | `b8bcc95ddd650d78c7458ca320bbd1e1295d610d55f03f919a55f0d04516cc32` |
| `P1/data/entities/train/source3/train_s3_entities.parquet` | 5,285,603 | 942,897,654 | `f1e0ace13b59ba2a84e696b2bc6fce2e65ce8fca219f6ac9368525582f9fc6f8` |
| `P1/data/entities/test/source1/test_s1_entities.parquet` | 1,732,544 | 307,175,551 | `e6ff58d72b00001e9078094ec427799294cbe478a256c50eb3795ad1c8a5578c` |
| `P1/data/entities/test/source2/test_s2_entities.parquet` | 4,887,273 | 904,777,638 | `62c74f7dfc62adc0497b88e663a6f4979e5138d370b72f0db829619aa5a73efe` |
| `P1/data/entities/test/source3/test_s3_entities.parquet` | 5,082,316 | 919,998,403 | `f7748097ca62b213977de22ba3a29764a9c9697d9ea40857dbd7d06f9dcbaa8c` |

---

## 3. Evaluated Candidate Artifacts & Checksums

| Candidate File Path | Rows | Size (bytes) | File SHA256 Checksum |
|---|---|---|---|
| `P2/data/candidates/train_candidate_pairs_s2.tsv` | 24,594,064 | 683,144,227 | `1a1f671d500bea88095cb69f5d32dabe6644357ff91432526305b10856f210be` |
| `P2/data/candidates/train_candidate_pairs_s3.tsv` | 29,998,661 | 833,639,183 | `b34f05adaefd6990495ebf2581f9d1b0582a15bb71a87d05a54c9caf6c440255` |
| `P2/data/candidates/train_candidate_pairs_s2_v2.tsv` | 30,359,040 | 843,272,201 | `386c0a0c0dacda2df90d26778a893c760b686c1c4e84bc0ef033e7ce84c28d22` |
| `P2/data/candidates/train_candidate_pairs_s3_v2.tsv` | 36,973,484 | 1,027,044,644 | `6d6cc5171beb4f445a4016f35a27e488e543cdeaa66bb5a26b342f87302c8404` |
| `P2/data/candidates/test_candidate_pairs_s2.tsv` | 30,232,352 | 779,219,819 | `7c79411c7e917cad04817adb921a721bda9a4927a32c01504599a9949ede8366` |
| `P2/data/candidates/test_candidate_pairs_s3.tsv` | 35,822,910 | 923,942,668 | `620ff6e2654bcd3eeae0e81b5670734b290fbc5a76d668891ad1659a7d606531` |
| `P2/data/candidates/test_candidate_pairs_s2_v2.tsv` | 34,919,169 | 900,172,941 | `149e168a56025a860c7e917f1ba70f990cfaefd0660175d8d8050c657fbe9c82` |
| `P2/data/candidates/test_candidate_pairs_s3_v2.tsv` | 41,714,627 | 1,075,341,054 | `ec2dd1c0ca63d71d242b84854b22bace3fec0c7d611a193fc224a6b6bf971576` |
| `outputs/person1_step1/test_candidate_pairs_s2.tsv` | 26,046,195 | 671,440,517 | `8aff61a3b98848ed3af607352f4def7e87cc2fb0abc927a9b6feb0d888586528` |
| `outputs/person1_step1/test_candidate_pairs_s3.tsv` | 30,777,878 | 793,767,104 | `645862ffab6e62ca0cf26d77d53b87148dc74be14e026b5918116c64bf3d829d` |

---

## 4. Phase 2 Produced Artifacts & Checksums

| Produced Artifact Path | Size (bytes) | SHA256 Checksum |
|---|---|---|
| `P1/scripts/evaluation/evaluate_candidate_set.py` | 13,000 | `f2720e1b971914826d84b01a0e7198c4af0e96b992536f7131edf7c301387f74` |
| `P1/scripts/evaluation/generate_phase2_reports.py` | 12,504 | `48f54105e2fbce853af591fe951a632369a913d4789f2e9ed8d4ef86b0e3fe33` |
| `P1/reports/PHASE2_CANDIDATE_BASELINE.tsv` | 5,335 | `0e9d12fd52f022061c5831a5d66b8fad3e5dbb6045e9a87db4c67d8dd5be75e8` |
| `P1/reports/PHASE2_CANDIDATE_SCHEMA_VALIDATION.tsv` | 1,646 | `c53871c11bd7a859704882fcd8a3cae7c6a1732a4d3b9735c1fbbd083999aa65` |
| `P1/reports/PHASE2_BASELINE_VALIDATION.md` | 10,750 | `ab8b4da8788f1268b67897bc4788ed7529b4886de8cae6ffc961ec2a4f23f2b7` |
| `P1/reports/PHASE3_ENTRY_CRITERIA.md` | 4,210 | `2e0c3050042a01683401cb7ca9672b754978c45fa8a974acf9e56e4aedd55137` |
