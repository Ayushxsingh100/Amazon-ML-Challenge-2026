# EXP-RET-01 — Soft Name Retrieval (Corporate Suffix Stripping)

**Experiment ID**: EXP-RET-01  
**Category**: Soft Name Retrieval  
**Date**: September 26, 2026  
**Status**: **ACCEPTED**

---

## 1. Hypothesis & Mechanism

**Hypothesis**: A significant fraction of V2 missed pairs (identified in Phase 1 analysis) fail exact name matching (Rule D/E) because legal entity suffix notations vary across sources (e.g. `Private Limited` vs `Pvt Ltd` vs `Pvtle Limited` vs `LLC` vs `Inc`). By deterministically stripping standard corporate and legal entity suffixes from the clean alphanumeric name (`name_clean`) while maintaining strict country blocking and a minimum core string length threshold ($\ge 5$), we can recover these true matches with controlled candidate volume.

### Exact Retrieval Rule
- **Blocking Key**: `country_normalized` + `core_name`
- **Definition of `core_name`**:
  ```sql
  regexp_replace(lower(name_clean), '(privatelimited|pvtlimited|pvtltd|pvtld|limited|ltd|llc|inc|corp|corporation|enterprises?|services?)$', '')
  ```
- **Filter Constraint**: `LENGTH(core_name) >= 5`

---

## 2. Experimental Results (Increment on V2 Baseline)

| Metric | V2 Baseline | EXP-RET-01 | Delta |
|---|---|---|---|
| **Total Candidates (Combined)** | 67,332,524 | 106,414,104 | +39,081,580 |
| **S2 Candidates** | 30,359,040 | 49,138,296 | +18,779,256 |
| **S3 Candidates** | 36,973,484 | 57,275,808 | +20,302,324 |
| **S2 True Captured** | 2,438,575 | 2,564,005 | **+125,430** |
| **S3 True Captured** | 2,584,593 | 2,708,133 | **+123,540** |
| **Combined True Captured** | 5,023,168 | 5,272,138 | **+248,970** |
| **S2 Recall** | 66.0213% | **69.4171%** | **+3.3958%** |
| **S3 Recall** | 65.5199% | **68.6516%** | **+3.1317%** |
| **Combined Recall** | 65.7623% | **69.0218%** | **+3.2595%** |
| **Remaining Missed Pairs** | 2,615,197 | 2,366,227 | -248,970 |
| **Zero-Candidate S1 Count** | 94,043 | **62,626** | **-31,417** |
| **p50 Fanout** | 5.0 | 7.0 | +2.0 |
| **p90 Fanout** | 81.0 | 130.0 | +49.0 |
| **p95 Fanout** | 155.0 | **269.0** | +114.0 ($\le 300$ **PASS**) |
| **p99 Fanout** | 346.0 | 585.0 | +239.0 |
| **p99.9 Fanout** | 850.0 | 920.0 | +70.0 |
| **Max Fanout** | 3,306 | 3,306 | +0 |

---

## 3. Decision & Trade-Off Analysis

- **Recall**: +3.2595% combined recall increase (recovering 248,970 real true ground truth pairs).
- **Fanout**: 95th percentile is 269.0 candidates per S1 entity, strictly satisfying the $p95 \le 300$ requirement. Max fanout did not grow (remained 3,306).
- **Zero-Candidate Reduction**: Reduced zero-candidate entities by **31,417**, activating previously unmatched entities.
- **Verdict**: **ACCEPTED**.
