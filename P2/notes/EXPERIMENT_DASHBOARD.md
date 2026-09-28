# Amazon ML Challenge 2026 — P2 Experiment Dashboard

## 1. Current Experiment Table

| Experiment | Candidate Version | OOF Macro F0.5 | Leaderboard | Status | Notes |
|---|---|---|---|---|---|
| **E02** | `cands_BCD_v1` (54.6M train) | 0.718765173 | — | Superseded | P3 authoritative internal baseline at fixed $T=0.50$; 10% negative downsampling recipe. |
| **E03** | `cands_BCD_v1` (54.6M train) | 0.744885727 | — | Verified (OOF) | Nested threshold optimization ($T^*=0.900$ across all 5 folds); zero retraining over E02 models. |
| **E06** | `cands_BCD_v2` (67.3M train) | 0.747085258 | **0.700398** | **Submitted (Live)** | 5-fold LightGBM ensemble on V2 candidate universe; submitted at fixed $T=0.50$. |

---

## 2. E06 Verified Configuration

- **Models**: 5 LightGBM Booster models (`P2/models/e06_v2_lgb_fold0.txt` through `P2/models/e06_v2_lgb_fold4.txt`)
  - Trees: 500 rounds | Max Leaves: 31 | Learning Rate: 0.05 | Seed: 2026
  - Features: 19 canonical features computed in-memory via DuckDB SQL
- **Candidate Generator**: [`P2/scripts/cands_BCD_v2_tasks.py`](file:///c:/NEW%20AMAZON/Amazon-ML-Challenge-2026/P2/scripts/cands_BCD_v2_tasks.py)
  - Rule Definitions (9 unioned rules): Rules A, B, C, D + Rule E (CleanName), Rule F1/F2 (HouseNorm zero-stripped), Rule G (RootToken+House), Rule H (CleanAddress), Rule I (ZipPrefix3)
- **Candidate Volume**:
  - **Train S2**: 30,359,040 pairs
  - **Train S3**: 36,973,484 pairs
  - **Train Total**: **67,332,524 pairs** (captured true pairs: 5,023,168 / 7,638,365; candidate recall: **65.7623%**)
  - **Test S2**: 34,919,169 pairs
  - **Test S3**: 41,714,627 pairs
  - **Test Total**: **76,633,796 pairs**
- **Decision Threshold**: $T = 0.50$ (arithmetic ensemble mean: `score = (p0 + p1 + p2 + p3 + p4) / 5.0`)
- **OOF Validation Status**: **Independently verified by P3**
  - Full Canonical OOF Rows: 67,332,524
  - Mean Macro F0.5: **`0.747085258`**
  - Standard Deviation: `0.000503748`
  - Fold Wins vs E02: **5 / 5 folds**
- **Candidate Provenance**: **Verified** (Git Commit SHA: `49a2f102e7161ff85f45e6c4be5c5b0ac0e3c210`)
- **Submission Artifacts**:
  - `output/matching_results.tsv` (1,732,544 rows; 9,928,563 predicted match pairs)
  - `output/candidate_pairs.tsv` (1,732,544 rows; 76,633,796 candidate pairs)
  - Submission Validator: **PASS** (`python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir data/test`)

---

## 3. Leaderboard vs OOF

| Metric Scope | Score (Macro F0.5) | Details |
|---|---|---|
| **E06 Full-OOF (Internal Cross-Validation)** | **`0.747085258`** | 5-fold cross-validation on 67,332,524 pairs across 2,206,821 S1 entities |
| **E06 Leaderboard (Public Evaluation)** | **`0.700398`** | Evaluated against hidden test ground truth (1,732,544 S1 entities) |
| **Observed Generalization Gap** | **`0.046687258`** | **~4.67 Macro F0.5 points** |

> [!NOTE]
> This gap is strictly an **observed empirical result**, NOT a diagnosed cause. Root cause analysis should be investigated systematically without speculative assumptions.

### Note on Candidate Recall Figures:
- **65.7623% Candidate Recall**: Applies to `cands_BCD_v2` (E06 candidate universe; 67,332,524 pairs; capturing 5,023,168 true pairs).
- **53.7289% Candidate Recall**: Applies to the legacy Strategy-B (V0) 2-rule artifact (`train_strategy_b_candidates_s2/s3.tsv`; 35,591,815 pairs) and must **NOT** be confused with or attributed to E06.

---

## 4. Current Bottleneck Questions

The following are **OPEN QUESTIONS** for systematic investigation, not conclusions:

1. **Generalization Gap**: Why does E06 generalize from 0.7471 internal OOF to 0.700398 on the public test leaderboard? (e.g., test country/distribution shift, France partition behavior, uncalibrated $T=0.50$ fixed threshold under negative downsampling).
2. **P1 Candidate Generation Pipeline**: What changes and structural improvements does P1's new ground-truth audit and candidate generation work introduce?
3. **New Candidate Ceiling**: What is the new Candidate Oracle Macro F0.5 ceiling once P1 delivers the updated candidate universe?
4. **Primary Gain Vector**: Is the next major score gain more effectively achieved through:
   - Candidate retrieval expansion (higher candidate recall ceiling)?
   - Decision rule optimization (nested threshold calibration, source-specific thresholds, entity-level pruning)?
   - Model capacity / feature engineering (interaction terms, TF-IDF / fuzzy tokens)?
   - Semantic retrieval (dense embeddings, multilingual cross-encoders)?
5. **Compute Strategy**: Is GPU / Colab compute justified by empirical evidence, or does CPU-optimized DuckDB + LightGBM remain sufficient and faster for iteration?

---

## 5. Team Ownership & Responsibilities

- **Person 1 (P1)**: Data normalization, ground-truth audit, candidate generation & blocking strategies.
- **Person 2 (P2)**: Feature engineering, model training, test-time inference, and modeling experiments.
- **Person 3 (P3)**: Validation protocol, fold governance, metrics scoring, loss decomposition, and final submission decisions.

---

## 6. Current Status

- **E06 Submission**: Successfully created, validated, and submitted.
- **Leaderboard Benchmark**: **`0.700398` Macro F0.5** established as our first live verified benchmark.
- **P3 E06 Validation**: Complete (0.747085258 OOF confirmed).
- **P3 Candidate Provenance Audit**: Complete (`cands_BCD_v2` lineage verified).
- **P1 Work**: Still in progress.
- **Next Experiment (E07+)**: **NOT DECIDED** (awaiting P1 handoff and team consensus).

---

## 7. Evidence / Artifact Paths

The facts and metrics in this dashboard are drawn directly from the following repository artifacts:

- **E06 Full Canonical OOF**: [`P2/reports/E06_V2_OOF_FULL_CANONICAL.tsv`](file:///c:/NEW%20AMAZON/Amazon-ML-Challenge-2026/P2/reports/E06_V2_OOF_FULL_CANONICAL.tsv)
- **E06 Interim Modeling Report**: [`P2/reports/E06_V2_INTERIM_REPORT.md`](file:///c:/NEW%20AMAZON/Amazon-ML-Challenge-2026/P2/reports/E06_V2_INTERIM_REPORT.md)
- **E06 Interim Results**: [`P2/reports/E06_V2_INTERIM_RESULTS.tsv`](file:///c:/NEW%20AMAZON/Amazon-ML-Challenge-2026/P2/reports/E06_V2_INTERIM_RESULTS.tsv)
- **E06 Modeling Pipeline**: [`P2/scripts/e06_v2_modeling.py`](file:///c:/NEW%20AMAZON/Amazon-ML-Challenge-2026/P2/scripts/e06_v2_modeling.py)
- **V2 Candidate Manifest**: [`P2/reports/cands_BCD_v2_manifest.tsv`](file:///c:/NEW%20AMAZON/Amazon-ML-Challenge-2026/P2/reports/cands_BCD_v2_manifest.tsv)
- **V2 Rule Ablation**: [`P2/reports/cands_BCD_v2_rule_ablation.tsv`](file:///c:/NEW%20AMAZON/Amazon-ML-Challenge-2026/P2/reports/cands_BCD_v2_rule_ablation.tsv)
- **V2 Candidate Generation Report**: [`P2/reports/CANDIDATE_GENERATION_V2_REPORT.md`](file:///c:/NEW%20AMAZON/Amazon-ML-Challenge-2026/P2/reports/CANDIDATE_GENERATION_V2_REPORT.md)
- **V2 Acceptance Report**: [`P2/reports/C01_V2_ACCEPTANCE_REPORT.md`](file:///c:/NEW%20AMAZON/Amazon-ML-Challenge-2026/P2/reports/C01_V2_ACCEPTANCE_REPORT.md)
- **E03 Nested Threshold Report**: [`P2/reports/E03_NESTED_THRESHOLD_REPORT.md`](file:///c:/NEW%20AMAZON/Amazon-ML-Challenge-2026/P2/reports/E03_NESTED_THRESHOLD_REPORT.md)
- **E02 Loss Decomposition**: [`P2/reports/E02_LOSS_DECOMPOSITION.md`](file:///c:/NEW%20AMAZON/Amazon-ML-Challenge-2026/P2/reports/E02_LOSS_DECOMPOSITION.md)
- **Fold Splitting Manifest**: [`P3/reports/folds_v1_manifest.tsv`](file:///c:/NEW%20AMAZON/Amazon-ML-Challenge-2026/P3/reports/folds_v1_manifest.tsv)
- **Legacy Strategy-B Recall**: [`P2/reports/strategy_b_train_recall.json`](file:///c:/NEW%20AMAZON/Amazon-ML-Challenge-2026/P2/reports/strategy_b_train_recall.json)
- **Submission Outputs**: [`output/matching_results.tsv`](file:///c:/NEW%20AMAZON/Amazon-ML-Challenge-2026/output/matching_results.tsv), [`output/candidate_pairs.tsv`](file:///c:/NEW%20AMAZON/Amazon-ML-Challenge-2026/output/candidate_pairs.tsv)
