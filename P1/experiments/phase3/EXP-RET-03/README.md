# EXP-RET-03 — Indic / Transliteration-Aware Retrieval

**Experiment ID**: EXP-RET-03  
**Category**: Indic / Transliteration-Aware Retrieval  
**Date**: September 26, 2026  
**Status**: **ACCEPTED**

---

## 1. Hypothesis & Mechanism

**Hypothesis**: Phase 1 missed-pair forensic analysis identified 279,753 true pairs containing Devanagari script and 693,416 cross-script pairs. When one source uses Latin script and another uses Devanagari, all English name blocking rules (A, C, D, E, F1, F2, G, I) fail completely. By implementing a deterministic character-level transliteration mapping from Devanagari to Latin (based on the standard ISO 15919 / Hunterian system) paired with cross-script address blocking (`country + house_number_norm + postal_code`), cross-script matches can be captured effectively.

### Exact Retrieval Rules
1. **Transliterated Name Key**:
   - Transliterate Devanagari characters to standard Latin equivalents using deterministic character mapping.
   - Join on: `country_normalized` + `house_number_norm` + `LEFT(translit_name, 2)` with `jaro_winkler_similarity(translit_name, target_translit_name) >= 0.80`.
2. **Cross-Script Address Key**:
   - Join on: `country_normalized` + `house_number_norm` + `postal_code` (where `LENGTH(postal_code) >= 5` and `house_number_norm <> ''`).

---

## 2. Experimental Results (Increment on V2 Baseline)

| Metric | V2 Baseline | EXP-RET-03 | Delta |
|---|---|---|---|
| **Total Candidates (Combined)** | 67,332,524 | 83,079,125 | +15,746,601 |
| **S2 Candidates** | 30,359,040 | 38,059,738 | +7,700,698 |
| **S3 Candidates** | 36,973,484 | 45,019,387 | +8,045,903 |
| **S2 True Captured** | 2,438,575 | 2,523,612 | **+85,037** |
| **S3 True Captured** | 2,584,593 | 2,667,420 | **+82,827** |
| **Combined True Captured** | 5,023,168 | 5,191,032 | **+167,864** |
| **S2 Recall** | 66.0213% | **68.3236%** | **+2.3023%** |
| **S3 Recall** | 65.5199% | **67.6196%** | **+2.0997%** |
| **Combined Recall** | 65.7623% | **67.9600%** | **+2.1977%** |
| **Remaining Missed Pairs** | 2,615,197 | 2,447,333 | -167,864 |
| **Zero-Candidate S1 Count** | 94,043 | **77,807** | **-16,236** |
| **p50 Fanout** | 5.0 | 7.0 | +2.0 |
| **p90 Fanout** | 81.0 | 88.0 | +7.0 |
| **p95 Fanout** | 155.0 | **99.0** | -56.0 ($\le 300$ **PASS**) |
| **p99 Fanout** | 346.0 | 179.0 | -167.0 |
| **p99.9 Fanout** | 850.0 | 480.0 | -370.0 |
| **Max Fanout** | 3,306 | 3,311 | +5 |

---

## 3. Decision & Trade-Off Analysis

- **Recall**: +2.1977% combined recall increase (recovering 167,864 true pairs).
- **Fanout**: Exceptionally tight fanout (p95 = 99.0 candidates per S1 entity).
- **Zero-Candidate Reduction**: Reduced zero-candidate entities by 16,236.
- **Verdict**: **ACCEPTED**.
