# EXP-RET-02 — Relaxed Name + House Retrieval (Threshold Sweep)

**Experiment ID**: EXP-RET-02  
**Category**: Relaxed Name + House Retrieval  
**Date**: September 26, 2026  
**Status**: **ACCEPTED** (Recommended Threshold: `JW >= 0.80`)

---

## 1. Hypothesis & Mechanism

**Hypothesis**: Phase 1 missed-pair forensic analysis identified that 967,093 true pairs (37.0% of all V2 misses) had exact matching normalized house numbers (`house_norm`), but failed because V2 name blocking required strict prefix4, token1, or root_token matching. By relaxing name similarity using Jaro-Winkler distance on `name_clean` constrained by `country_normalized` + `house_number_norm` + `prefix2(name_clean)`, we recover true matches where names suffered from minor spelling variations, word order changes, or abbreviations without combinatorial explosion.

### Exact Retrieval Rule
- **Blocking Key**: `country_normalized` + `house_number_norm` + `LEFT(name_clean, 2)`
- **Similarity Constraint**: `jaro_winkler_similarity(s1.name_clean, tgt.name_clean) >= THRESHOLD`
- **Filter Constraint**: `house_number_norm <> '' AND LENGTH(name_clean) >= 3`

---

## 2. Multi-Threshold Sweep Evaluation

All thresholds evaluated against real train ground truth incrementally on top of V2 baseline:

| Threshold | Total Candidates | S2 Recall | S3 Recall | Combined Recall | Recall Delta | New True Captures | Zero S1 | p50 | p95 | p99 | Max |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **V2 Baseline** | 67,332,524 | 66.0213% | 65.5199% | 65.7623% | — | — | 94,043 | 5.0 | 155.0 | 346.0 | 3,306 |
| **JW >= 0.85** | 68,350,016 | 67.1370% | 66.7818% | 66.9536% | +1.1913% | +90,989 | 89,621 | 6.0 | 82.0 | 157.0 | 3,306 |
| **JW >= 0.80** | **75,999,866** | **67.2253%** | **66.8850%** | **67.0496%** | **+1.2873%** | **+98,323** | **83,084** | **6.0** | **90.0** | **173.0** | **3,311** |
| **JW >= 0.75** | 93,170,553 | 67.2423% | 66.9094% | 67.0704% | +1.3081% | +99,911 | 74,687 | 7.0 | 104.0 | 206.0 | 3,318 |

---

## 3. Threshold Selection & Trade-Off Analysis

- **0.85 vs 0.80**: Moving from 0.85 to 0.80 captures **7,334 additional true pairs** and reduces zero-candidate S1s by **6,537** with a moderate candidate increase (+7.6M pairs) and tight fanout (p95 = 90.0).
- **0.80 vs 0.75**: Moving from 0.80 to 0.75 adds **17,170,687 candidates** to capture only **1,588 additional true pairs** (marginal precision of only 0.009%).
- **Conclusion**: **`JW >= 0.80`** provides the optimal balance of recall gain and fanout control.
- **Verdict**: **ACCEPTED** at threshold 0.80.
