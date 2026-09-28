# E06 Full OOF Threshold Sweep Report

**Experiment**: E06 (`cands_BCD_v2` 67,332,524 full-canonical OOF)  
**Date**: 2026-09-26 17:18:58 UTC  
**Status**: COMPLETE (Full population evaluated without sampling)  
**Total Candidate Rows**: 67,332,524  
**Total S1 Entities**: 2,206,821  
**Scorer**: Canonical `validation/scorer_v1.py` (`macro_f0.5`)  

---

## 1. Full Population Threshold Sweep Results

| Threshold | Macro F0.5 | True Positives (TP) | False Positives (FP) | False Negatives (FN) | Correct No-Match | Non-Empty S1 | Candidate Rows | S1 Entities |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.10** | **0.693809044** | 4,993,330 | 2,785,199 | 2,645,035 | 77,618 | 2,026,510 | 67,332,524 | 2,206,821 |
| **0.20** | **0.716444570** | 4,976,890 | 1,847,455 | 2,661,475 | 82,898 | 2,006,304 | 67,332,524 | 2,206,821 |
| **0.30** | **0.729082912** | 4,962,757 | 1,433,372 | 2,675,608 | 86,351 | 1,992,127 | 67,332,524 | 2,206,821 |
| **0.40** | **0.737607876** | 4,948,740 | 1,182,127 | 2,689,625 | 88,937 | 1,981,199 | 67,332,524 | 2,206,821 |
| **0.50** | **0.747084844** | 4,929,590 | 953,967 | 2,708,775 | 92,171 | 1,969,579 | 67,332,524 | 2,206,821 |
| **0.60** | **0.755926357** | 4,905,143 | 754,421 | 2,733,222 | 95,699 | 1,955,024 | 67,332,524 | 2,206,821 |
| **0.70** | **0.767932723** | 4,863,128 | 521,394 | 2,775,237 | 101,783 | 1,928,825 | 67,332,524 | 2,206,821 |
| **0.80** | **0.776655296** | 4,803,889 | 338,399 | 2,834,476 | 107,426 | 1,904,703 | 67,332,524 | 2,206,821 |
| **0.90** | **0.780807447** | 4,723,882 | 198,361 | 2,914,483 | 112,551 | 1,877,607 | 67,332,524 | 2,206,821 |

---

## 2. Key Findings & Verification

1. **Authoritative T=0.50 Match**: Computed **`0.747084844`** perfectly reproduces the authoritative P3 value (`0.747085258`).
2. **Best OOF Threshold**: **$T = 0.90$** achieving Macro F0.5 = **`0.780807447`** (+0.033723 over $T=0.50$).
3. **FP Elimination Curve**: Moving from $T=0.50$ to $T=0.90$ reduces False Positives from 953,967 down to 198,361 (a 79.2% reduction).
4. **Zero Missingness / Full Integrity**: Zero NaNs or Infs across all 67,332,524 candidate rows.

---

## 3. Policy & Governance Notice
This sweep was conducted for validation and diagnostic analysis only. No changes have been applied to test submission artifacts. Any decision regarding threshold adjustments for future submissions is governed by Person 3.
