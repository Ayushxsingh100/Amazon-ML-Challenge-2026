# P1 Data Directory

## Large Dataset Referencing Strategy
To avoid inflating the repository with redundant copies of massive TSV files (>10 GB):
1. **Raw Datasets**: Authoritative files reside in `data/train/` and `data/test/`.
2. **Normalized Datasets**: Authoritative normalized files reside in `outputs/person1_step1/normalized/`.
3. **Ground Truth**: Authoritative ground truth file is `data/train/train_ground_truth.tsv`.
4. **Candidate Pairs**:
   - V1 baseline candidates reside in `P2/data/candidates/train_candidate_pairs_s{2,3}.tsv`.
   - V2 candidates reside in `P2/data/candidates/train_candidate_pairs_s{2,3}_v2.tsv`.
   - V3 candidates directory (`P1/data/candidates/v3/`) is a placeholder for future implementation.

All file paths, byte sizes, schemas, and SHA256 hashes are strictly tracked in:
- `P1/manifests/data_manifest.tsv`
- `P1/manifests/artifact_manifest.tsv`
- `P1/checksums/SHA256SUMS.txt`
- `P1/configs/phase0_baseline.yaml`
