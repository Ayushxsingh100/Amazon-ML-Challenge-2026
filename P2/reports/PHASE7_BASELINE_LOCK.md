# Amazon ML Challenge 2026 — Phase 7 Baseline Lock
## Verified Current Production Baseline Before Phase 7 Modifications

- **Date:** 2026-09-27
- **Base Git Commit:** `eae882c8be1a4f1e17ca9ad824b6b8fe657ba009`
- **Active Working Branch:** `phase7`
- **Current Verified Leaderboard Score:** **~0.764**
- **Previous Leaderboard Score:** **~0.702** (Phase 5 uncalibrated release)

---

## 1. Locked Production Artifacts & Checksums

| Artifact | File Path | Records / Rows | File Size (Bytes) | SHA256 Checksum |
| :--- | :--- | :--- | :--- | :--- |
| **Current Submission** | `output/matching_results.tsv` | 1,732,544 data rows (1,732,545 lines) | 81,238,977 | `fce26bfc78d6c2a74829ff512981d94d3b66d89ca5e1d232b14ea1eafa402f9a` |
| **Preserved Baseline Copy** | `P2/predictions/phase7/baseline_matching_results.tsv` | 1,732,544 data rows (1,732,545 lines) | 81,238,977 | `fce26bfc78d6c2a74829ff512981d94d3b66d89ca5e1d232b14ea1eafa402f9a` |
| **Test S2 Candidate Pairs** | `P1/data/candidates/v3/test_candidate_pairs_s2_v3.tsv` | 43,841,928 | 1,510,836,431 | `1a4e94cc9b15ba984588db2b763e6c15967b2d7a7ef04213833d79dfe11307af` |
| **Test S3 Candidate Pairs** | `P1/data/candidates/v3/test_candidate_pairs_s3_v3.tsv` | 51,354,667 | 1,774,881,395 | `503282181cff0df65b86f3f19c2f2813b4cd84d99a1a2b4ed167ac2199d5e3aa` |
| **Train S2 Candidate Pairs** | `P1/data/candidates/v3/train_candidate_pairs_s2_v3.tsv` | 44,792,028 | 1,544,142,654 | Canonical V3 train S2 candidates |
| **Train S3 Candidate Pairs** | `P1/data/candidates/v3/train_candidate_pairs_s3_v3.tsv` | 48,379,921 | 1,673,348,683 | Canonical V3 train S3 candidates |
| **Fold 0 Model Booster** | `P2/models/phase4/lgb_fold0.txt` | 100 trees | 2,741,403 | Preserved LightGBM booster |
| **Phase 4 Boosters [0-4]** | `P2/models/phase4/lgb_fold*.txt` | 5 models | ~13.7 MB total | Preserved 5-fold ensemble |

---

## 2. Locked Baseline Measured Metrics

- **Current Leaderboard Score:** `~0.764` (Verified live competition submission)
- **Previous Leaderboard Score:** `~0.702`
- **Total Test Candidates Universe:** `95,196,595` (S2: 43,841,928 + S3: 51,354,667)
- **Total Train Candidates Universe:** `93,171,949` (S2: 44,792,028 + S3: 48,379,921)
- **Total Train Ground Truth Pairs:** `7,638,365`
- **Retrieved Train Ground Truth Pairs:** `5,511,986`
- **Missed Train Ground Truth Pairs:** `2,126,379`
- **Candidate Recall (V3):** `72.161857%`
  - S2 Candidate Recall: `72.4818%` (2,677,201 / 3,693,619)
  - S3 Candidate Recall: `71.8623%` (2,834,785 / 3,944,746)
- **Fold 0 Macro $F_{0.5}$ (Full Negative Universe):** `0.805917`
- **Fold 0 Micro Precision:** `0.9595`
- **Fold 0 Micro Recall:** `0.6493`
- **Singleton Accuracy:** `0.981249` (43,203 / 44,029 correct singletons on Fold 0)
- **Final Test Matched Entity Count:** `1,539,711` (88.87%)
- **Final Test No-Match Singletons:** `192,833` (11.13%)
- **Final Test Total Prediction Count (Instances):** `4,555,480`
- **Phase 6 Calibrated Decision Policy:** Threshold $T = 0.86$, Score Margin $\Delta \le 0.06$, Max Target Cap $K \le 8$

---

## 3. Preserved Baseline Integrity Rule

Under no circumstances may `P2/predictions/phase7/baseline_matching_results.tsv` or `output/matching_results.tsv` be modified until an experimental pipeline is demonstrably and measurably superior on the competition metric across cross-validation folds.
