# E06 Fine-Grained OOF Threshold Sweep Report (0.90 - 0.99)

**Experiment**: E06 (`cands_BCD_v2` 67,332,524 full-canonical OOF)  
**Date**: 2026-09-26 17:29:12 UTC  
**Status**: COMPLETE (Full population evaluated without sampling)  
**Total Candidate Rows**: 67,332,524  
**Total S1 Entities**: 2,206,821  
**Scorer**: Canonical `validation/scorer_v1.py` (`macro_f0.5`)  

---

## 1. Fine-Grained Threshold Sweep Results

| Threshold | Macro F0.5 | True Positives (TP) | False Positives (FP) | False Negatives (FN) | Correct No-Match | Non-Empty S1 | Empty S1 | Candidate Rows | S1 Entities |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.90 *** | **0.780807447** | 4,723,882 | 198,361 | 2,914,483 | 112,551 | 1,877,607 | 329,214 | 67,332,524 | 2,206,821 |
| **0.91** | **0.780798266** | 4,712,024 | 185,501 | 2,926,341 | 113,069 | 1,874,326 | 332,495 | 67,332,524 | 2,206,821 |
| **0.92** | **0.780708592** | 4,698,250 | 171,917 | 2,940,115 | 113,645 | 1,870,683 | 336,138 | 67,332,524 | 2,206,821 |
| **0.93** | **0.780377953** | 4,681,122 | 157,634 | 2,957,243 | 114,280 | 1,866,371 | 340,450 | 67,332,524 | 2,206,821 |
| **0.94** | **0.779773864** | 4,659,720 | 142,006 | 2,978,645 | 114,960 | 1,861,247 | 345,574 | 67,332,524 | 2,206,821 |
| **0.95** | **0.778737556** | 4,632,231 | 125,482 | 3,006,134 | 115,793 | 1,854,933 | 351,888 | 67,332,524 | 2,206,821 |
| **0.96** | **0.776818126** | 4,594,578 | 107,664 | 3,043,787 | 116,655 | 1,846,592 | 360,229 | 67,332,524 | 2,206,821 |
| **0.97** | **0.773234640** | 4,538,534 | 87,869 | 3,099,831 | 117,646 | 1,834,655 | 372,166 | 67,332,524 | 2,206,821 |
| **0.98** | **0.766295206** | 4,446,531 | 65,899 | 3,191,834 | 118,864 | 1,816,042 | 390,779 | 67,332,524 | 2,206,821 |
| **0.99** | **0.748055867** | 4,242,177 | 40,368 | 3,396,188 | 120,489 | 1,776,371 | 430,450 | 67,332,524 | 2,206,821 |

---

## 2. Key Findings & Peak Analysis

1. **Optimal Threshold**: **$T = 0.90$** achieves peak Macro F0.5 = **`0.780807447`**.
2. **Gain vs $T=0.50$ Baseline**: +0.033723 Macro F0.5 points.
3. **Gain vs $T=0.90$**: +0.000000 Macro F0.5 points.
4. **Error Trade-off at Peak**:
   - True Positives: 4,723,882 (retaining 94.04% of captured candidate positives)
   - False Positives: 198,361 (reduced from 953,967 at $T=0.50$, an 79.2% reduction in false alarms)
   - Correct No-Match S1: 112,551 / 123,247 (91.32% true negative accuracy)

---

## 3. Governance Notice
This sweep was performed for P3 threshold selection and diagnostic analysis only. No changes have been applied to competition submissions.
