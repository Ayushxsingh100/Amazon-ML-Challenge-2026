# Phase 6 Baseline Lock

**Date**: September 27, 2026  
**Repository**: `Amazon-ML-Challenge-2026`  
**Purpose**: Immutable baseline record prior to Phase 6 forensic diagnosis and optimization.

---

## 1. Verified Production Artifacts & Hashes

| Artifact Description | File Path | Rows / Volume | SHA256 Checksum |
|:---|:---|---:|:---|
| **Current Production Submission** | `output/matching_results.tsv` | 1,732,544 S1 rows | `c0d5da59dfafabd771ca5fbd1dba655b41fff458af9ff470e3a2ecb86535bbbc` |
| **Preserved Baseline Backup** | `P2/predictions/phase6/baseline_matching_results.tsv` | 1,732,544 S1 rows | `c0d5da59dfafabd771ca5fbd1dba655b41fff458af9ff470e3a2ecb86535bbbc` |
| **Phase 4 S2 Test Predictions** | `P2/predictions/phase4/test_predictions_s2.tsv` | 43,841,928 rows | `1a4e94cc9b15ba984588db2b763e6c15967b2d7a7ef04213833d79dfe11307af` |
| **Phase 4 S3 Test Predictions** | `P2/predictions/phase4/test_predictions_s3.tsv` | 51,354,667 rows | `503282181cff0df65b86f3f19c2f2813b4cd84d99a1a2b4ed167ac2199d5e3aa` |
| **Phase 3 V3 Train S2 Candidates** | `P1/data/candidates/v3/train_candidate_pairs_s2_v3.tsv` | 42,933,945 pairs | `61d5dce6389bf424daf61f842ed0f41ebc3991fa27763d4641c4ab4931a7508c` |
| **Phase 3 V3 Train S3 Candidates** | `P1/data/candidates/v3/train_candidate_pairs_s3_v3.tsv` | 50,238,004 pairs | `d5e90bb6d9a5d81154a83366d98211352e95bebcbc84c5011f14cd0bfe10eeeb` |
| **Phase 3 V3 Test S2 Candidates** | `P1/data/candidates/v3/test_candidate_pairs_s2_v3.tsv` | 43,841,928 pairs | `4aa0c71ea70a98aff77088da2c4b070e8fb305b8fbebfbda4daa4fa14398dfa2` |
| **Phase 3 V3 Test S3 Candidates** | `P1/data/candidates/v3/test_candidate_pairs_s3_v3.tsv` | 51,354,667 pairs | `f24041fbbc3a5f56a4b9c26b20dfde3377770f85427a9373bc9c1efc8bf441cd` |
| **Train Ground Truth** | `outputs/person1_step1/train_ground_truth_reconstructed.tsv` | 2,206,821 S1s (7,638,365 pairs) | `70bc1d8a16c667e0155c2105d0ab2ebe41d7e7a85d8a529e3ca81c6c3a5af037` |
| **Canonical Folds Manifest** | `P3/reports/folds_v1_manifest.tsv` | 2,206,821 rows | `9dcec5d83a477d224067e71b93abc21af8befa26f9c399568121fa83ba8801a3` |

---

## 2. Baseline Performance Benchmarks

- **Current Public Leaderboard Result**: approximately **`0.702`** (previous baseline was ~`0.700`).
- **Production V3 Candidate Retrieval Recall**:
  - Train S2 Recall: `72.481785%` (2,677,201 / 3,693,619 true pairs)
  - Train S3 Recall: `71.862295%` (2,834,785 / 3,944,746 true pairs)
  - **Combined Train Candidate Recall**: **`72.161857%`** (5,511,986 / 7,638,365 true pairs)
  - Candidate Ceiling Loss: `27.838143%` of true relationships are completely absent from candidates.
- **Phase 4 Recorded Out-of-Fold (OOF) Metrics @ T=0.60**:
  - Sample Evaluated: 14,282,556 pairs (100% positives + 10% downsampled hash negatives)
  - Pairwise Precision: `0.979470`
  - Pairwise Recall: `0.964903`
  - Pairwise $F_{0.5}$: `0.976527`
  - **Recorded OOF Macro $F_{0.5}$ (on 10% negative sample)**: `0.841781`
- **Current Final Submission Metrics (`output/matching_results.tsv`)**:
  - Total S1 Entities Evaluated: `1,732,544` (100% test coverage)
  - Total Matched Pairs Submitted: `10,831,951`
  - S1 Entities with $\ge 1$ Matched Target: `1,600,875` (92.40%)
  - S1 Entities with 0 Matched Targets (No-Match): `131,669` (7.60%)
  - S1 Entities with Exactly 1 Target: `241,057` (13.91%)
  - S1 Entities with $>1$ Targets: `1,359,818` (78.49%)

---

## 3. The Core Discrepancy to Solve

The offline 5-fold cross-validation reported Macro $F_{0.5} = 0.841781$, but the live competition leaderboard score is only ~`0.702`.
Investigation reveals that the Phase 4 OOF validation was computed strictly on a 10% downsampled negative sample (`df`, 14.3M rows), where the negative-to-positive ratio was 1.6 : 1. In full test inference across all 95.2M candidates, the negative-to-positive ratio is ~16 : 1. Scoring 100% of negative candidates at $T = 0.60$ causes an explosion of false positives (~10.83M submitted pairs vs ~4.33M true captured pairs), resulting in catastrophic precision collapse under the precision-heavy Macro $F_{0.5}$ metric.
