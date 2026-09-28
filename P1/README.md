# Person 1 — Data & Candidate Generation

## Responsibility
P1 owns:
- raw data verification
- reconstruction verification
- normalization
- entity dataset preparation
- candidate generation
- candidate recall
- candidate integrity
- reproducibility
- P1 → P2 handoff

## Current Phase
Phase 0 — Foundation & Baseline Freeze

## Important Rule
P1 candidate generation must never use ground truth to generate candidates.
Ground truth is for evaluation only.

## Current Baseline

### Verified Values (VERIFIED_NOW)
- **Train Source 1 Entities**: 2,206,821 rows, 2,206,821 unique IDs (0 duplicates, 0 nulls)
- **Train Source 2 Entities**: 5,034,616 rows, 5,034,616 unique IDs (0 duplicates, 0 nulls)
- **Train Source 3 Entities**: 5,285,603 rows, 5,285,603 unique IDs (0 duplicates, 0 nulls)
- **Test Source 1 Entities**: 1,732,544 rows, 1,732,544 unique IDs (0 duplicates, 0 nulls)
- **Test Source 2 Entities**: 4,887,273 rows, 4,887,273 unique IDs (0 duplicates, 0 nulls)
- **Test Source 3 Entities**: 5,082,316 rows, 5,082,316 unique IDs (0 duplicates, 0 nulls)
- **Ground Truth Pairs**: 7,638,365 exploded true match pairs (3,693,619 S2 + 3,944,746 S3)
- **Ground Truth Entities**: 2,206,821 S1 entities (123,247 no-match, 5.5848% structural floor)
- **V1 Train Candidate Volume**: 54,592,725 pairs (24,594,064 S2 + 29,998,661 S3)
- **V1 S2 Candidate Recall**: 59.9449% (2,214,137 / 3,693,619 captured)
- **V1 S3 Candidate Recall**: 59.4827% (2,346,442 / 3,944,746 captured)
- **V2 Train Candidate Volume**: 67,332,524 pairs (30,359,040 S2 + 36,973,484 S3)
- **V2 S2 Candidate Recall**: 66.0213% (2,438,575 / 3,693,619 captured)
- **V2 S3 Candidate Recall**: 65.5199% (2,584,593 / 3,944,746 captured)
- **E02 OOF Macro F0.5 (T=0.5)**: 0.769452 (evaluated on V1 candidates across 5 folds)
- **V1 Candidate Oracle Macro F0.5**: 0.784523

## Canonical Artifacts
- **Raw Train Ground Truth**: `data/train/train_ground_truth.tsv` (SHA256: `70bc1d8a16c667e0155c2105d0ab2ebe41d7e7a85d8a529e3ca81c6c3a5af037`)
- **Raw Sources**: `data/train/train_source{1,2,3}.tsv` and `data/test/test_source{1,2,3}.tsv`
- **Normalized Datasets**: `outputs/person1_step1/normalized/{train,test}_source{1,2,3}_normalized.tsv`
- **Frozen Folds Manifest**: `P3/reports/folds_v1_manifest.tsv` (SHA256: `9dcec5d83a477d224067e71b93abc21af8befa26f9c399568121fa83ba8801a3`)
- **Authoritative Scorer**: `validation/scorer_v1.py` & `validation/metrics.py`
- **V1 Candidate Artifacts**: `P2/data/candidates/train_candidate_pairs_s{2,3}.tsv`
- **V2 Candidate Artifacts**: `P2/data/candidates/train_candidate_pairs_s{2,3}_v2.tsv`
- **Canonical OOF Predictions**: `P2/reports/E02_OOF_FULL_CANONICAL_fold{0,1,2,3,4}.tsv`

## Current Known Issues
1. **Candidate Recall Ceiling**: Current V1 baseline candidates capture only ~59.7% of true pairs, capping pipeline performance at ~0.7845 F0.5.
2. **V2 Candidates Unused**: V2 candidate sets (capturing 66.0% / 65.5% recall, +462,589 true pairs) exist on disk in `P2/data/candidates/` but have not been trained on by the model.
3. **Train/Test Candidate Mismatch**: Test candidate scoring scripts reference stale Strategy B candidates (`outputs/person1_step1/test_candidate_pairs_s*.tsv`) which only use Rules A+B (26.0M/30.8M pairs), omitting Rules C+D (V1) and Rules E-I (V2).
4. **Missing Production Model Binaries**: LightGBM model binary files (`.lgb` / `.pkl`) are not stored locally; only the 5 full OOF prediction shards exist locally.
5. **Missing Test Submission**: No final `matching_results.tsv` submission file exists in the repository.

## Phase Status
Phase 0: COMPLETED
