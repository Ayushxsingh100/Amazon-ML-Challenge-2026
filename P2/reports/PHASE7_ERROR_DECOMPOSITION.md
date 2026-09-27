# Amazon ML Challenge 2026 — Phase 7 Error Decomposition
## Empirical Funnel Diagnostic & Error Classification

- **Date:** 2026-09-27
- **Phase:** Phase 7 (Candidate Recall Optimization & 5-Fold Grouped Validation)
- **Evaluation Split:** Cross-Validation Fold 0 (440,272 S1 entities, 1,525,486 ground truth pairs)
- **Winning Pipeline:** Candidate Set V4 + 5-Fold LightGBM + Policy ($T=0.88$, Margin=$0.05$, $K \le 8$)

---

## 1. The Core Funnel Diagnostic (Section 30)

The table below tracks the complete attrition of ground truth relationships from initial data through candidate generation, model ranking, and final decision thresholding:

| Pipeline Stage | Ground Truth Pairs Remaining | % of Total Ground Truth | Cumulative Attrition | Major Mechanism of Loss |
| :--- | :--- | :--- | :--- | :--- |
| **1. Ground Truth Total** | **1,525,486** | **100.00%** | 0 | Ground truth universe on Fold 0 |
| **2. Candidate-Retrieved GT (V4)** | **1,153,875** | **75.64%** | -371,611 (-24.36%) | Retrieval failure (unindexed cross-script text & severe typos) |
| **3. Model-Ranked GT ($Rank \le 8$)** | **1,098,420** | **72.00%** | -55,455 (-3.64%) | Ranking failure (true target outranked by false candidates) |
| **4. Decision-Policy Accepted GT** | **1,025,780** | **67.24%** | -72,640 (-4.76%) | Thresholding rejection ($score < 0.88$ or margin $> 0.05$) |
| **5. Final Predictions Submitted** | **1,075,876** | — | — | Includes 1,025,780 TPs and 50,096 FPs (**95.34% precision**) |

### Key Funnel Insights:
1. **Candidate Retrieval remains the largest loss component (24.36% lost):** Even after increasing recall from 72.16% to 75.64%, the retrieval stage accounts for **74.4% of all lost ground truth**.
2. **Model Ranking is exceptionally strong:** The model successfully ranks the true target in the top-8 for **95.2% of all retrievable candidates** (1,098,420 / 1,153,875).
3. **Decision Policy achieves near-optimal precision/recall trade-off:** By rejecting 72,640 low-confidence pairs, it secures a 95.34% micro precision on Fold 0, which maximizes Macro $F_{0.5}$.

---

## 2. Exhaustive Error Decomposition (Section 29)

Every error observed on Fold 0 is classified into distinct categories:

| Error Category | Error Count (Instances) | % of Total Errors | Impact on Macro $F_{0.5}$ | Root Cause & Description |
| :--- | :--- | :--- | :--- | :--- |
| **A. Retrieval Miss (Absent Candidate)** | 371,611 | 67.59% | High FN penalty | True target pair was never retrieved into candidate pool V4. |
| **B. Threshold Rejection (Low Model Score)** | 72,640 | 13.21% | Moderate FN penalty | True target was present, but scored below $T=0.88$ or had score margin $>0.05$. |
| **C. Ranking Failure (Outranked by FP)** | 55,455 | 10.09% | Combined FP/FN penalty | True target was present, but ranked $>8$ due to higher-scoring false candidates. |
| **D. Spurious False Positives (Partial Dilution)** | 45,940 | 8.36% | High FP penalty | Correct target was chosen, but an extra spurious false candidate was admitted. |
| **E. Singleton False Positives** | 4,156 | 0.76% | Severe FP penalty | True no-match S1 had $\ge 1$ candidate admitted by the model. |
| **TOTAL ERRORS** | **549,802** | **100.00%** | — | — |

---

## 3. Source-Specific Error Breakdown

| Metric | Source 2 (`S1-S2`) | Source 3 (`S1-S3`) | S2 vs S3 Comparison |
| :--- | :--- | :--- | :--- |
| **Total GT Pairs (Fold 0)** | 738,124 | 787,362 | S3 is slightly larger |
| **Retrieved GT Pairs (V4)** | 559,230 | 594,645 | S2: 75.76% recall, S3: 75.52% recall |
| **Accepted True Matches** | 497,820 | 527,960 | S2: 67.44%, S3: 67.05% |
| **False Positives** | 23,410 | 26,686 | S2: 95.51% prec, S3: 95.19% prec |

The error distribution between S2 and S3 is remarkably symmetrical across all stages of the pipeline.
