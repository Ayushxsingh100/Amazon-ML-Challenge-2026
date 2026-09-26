# Phase 3 Final Report: Candidate Retrieval Improvement & Production V3 Promotion

**Project**: Amazon ML Challenge 2026 — Entity Resolution
**Author**: Person 1 (P1)
**Date**: September 27, 2026
**Status**: **ACCEPTED & VALIDATED**

---

## 1. Scope & Objectives
Phase 3 focused exclusively on improving candidate retrieval recall beyond the frozen V2 baseline while enforcing strict computational fanout bounds on both Train and Test splits.

### Mandatory Phase 3 Gate Criteria
1. **TRAIN Combined Recall**: Must strictly exceed the V2 baseline of **65.7623%**.
2. **TEST Combined Fanout**: 95th percentile ($p95$) must be **$\le 300.0$** across all 1,732,544 canonical Test $S_1$ entities.
3. **Data Integrity**: Zero duplicates, zero null/malformed entity IDs, zero self-matches, and zero cross-country candidate pairs.
4. **Reproducibility**: Complete determinism, zero synthetic data, zero modifications to canonical entity Parquets or frozen ground truth.

---

## 2. Baseline Summary (V1 and V2)

### V1 Baseline
- **Retrieval Rules**: Rules A, B, C, D (Exact Address, Prefix4 + House, Token Overlap, Phone Number).
- **Train Recall**: 60.1081% (4,591,274 / 7,638,365 pairs captured).
- **Candidates**: 54,635,930 train pairs.

### Frozen V2 Baseline
- **Retrieval Rules**: Rules A through I (adds Reg/Tax IDs, Normalized Business Name Prefix, and City/Locality blocks).
- **Train Combined Recall**: **65.762346%** (5,023,168 / 7,638,365 pairs captured).
  - $S_2$ Recall: 65.9996% (2,437,775 / 3,693,619)
  - $S_3$ Recall: 65.5398% (2,585,393 / 3,944,746)
- **Train Fanout**: $p50 = 6.0, p90 = 104.0, p95 = 143.0, p99 = 277.0, \max = 1,894$.
- **Test Fanout**: $p50 = 6.0, p90 = 122.0, p95 = 256.0, p99 = 515.0, \max = 3,707$.
- **Test Candidate Volume**: 76,633,796 candidate pairs ($S_2$: 34,919,169; $S_3$: 41,714,627).
- **Zero-Candidate $S_1$ Entities**: 94,043 on Train (4.26%); 72,622 on Test (4.19%).

---

## 3. Original Uncapped V3 & Fanout Gate Discrepancy

Initial integration of four experimental retrieval rules (EXP-RET-01, 02, 03, 04) yielded substantial recall improvements on Train but breached the Phase 3 Test fanout gate:

| Split | Metric | Measured Uncapped V3 | Phase 3 Requirement | Gate Status |
|:---|:---|---:|---:|:---:|
| **TRAIN** | Combined Recall | **72.4229%** | $> 65.7623\%$ | **PASS** |
| **TRAIN** | Combined $p95$ | **199.0** | $\le 300.0$ | **PASS** |
| **TEST** | Combined $p95$ | **318.0** | $\le 300.0$ | **FAIL** |
| **TEST** | Candidate Pairs | 104,705,486 | Controlled | Monitored |
| **TEST** | Zero-Candidate $S_1$ | 33,407 | Minimize | Recaptured 39,215 entities |

The initial uncapped V3 artifacts failed the literal Phase 3 entry requirement ($p95 \le 300$). These artifacts were moved to [`P1/experiments/phase3/UNCAPPED_V3/`](file:///Users/krishnagera/Amazon-ML-Challenge-2026/P1/experiments/phase3/UNCAPPED_V3/) and preserved for audit.

---

## 4. Retrieval Experiment Evaluations (EXP-RET-01 to 04)

Each candidate generation rule was implemented and independently evaluated on top of V2:

### EXP-RET-01: Soft Name Retrieval
- **Rule**: Exact match on Country + Normalized Core Name (legal suffixes stripped) with `len(core_name) >= 5` and deterministic $S_1$ fanout cap $\le 100$.
- **Standalone Gain on V2**: **+5.0763 percentage points** (Recall: 70.8386%, Captured: 5,410,951 pairs).
- **Impact**: Primary driver of recall expansion. Decreased zero-candidate $S_1$ count by 51,430 entities.
- **Tail Impact**: Capped at $\le 100$, contributes only 0 to 3 candidates per entity to the extreme tail ($> 2,500$ fanout).

### EXP-RET-02: Relaxed Name + House Retrieval
- **Rule**: Exact match on Country + `house_number_norm` + `name_prefix_2` with Jaro-Winkler similarity $\ge 0.80$.
- **Standalone Gain on V2**: **+0.6728 percentage points** (Recall: 66.4351%, Captured: 5,074,561 pairs).
- **Tail Impact**: Uncapped in initial V3. Contributed 1,000–2,700 candidates to dense address blocks (e.g. house number '1', '2').

### EXP-RET-03: Indic / Transliteration-Aware Retrieval
- **Rule**: Devanagari ISO 15919 transliteration + `house_norm` + transliterated `name_prefix_2` (JW $\ge 0.80$) plus India cross-script address block (`house_norm` + `postal_code`).
- **Standalone Gain on V2**: **+0.2882 percentage points** (Recall: 66.0505%, Captured: 5,045,183 pairs).
- **Tail Impact**: On non-Devanagari entities, transliteration is identity, causing unconstrained duplication of EXP-RET-02 address matches on high-density house numbers.

### EXP-RET-04: Locality + Name Retrieval
- **Rule**: Country + address prefix 10 + name prefix 3, unioned with Country + postal code ($\ge 5$) + root token ($\ge 4$).
- **Standalone Gain on V2**: **+0.8016 percentage points** (Recall: 66.5639%, Captured: 5,084,395 pairs).
- **Tail Impact**: Highly selective; adds only +10 fanout units to V2 test $p95$ (256.0 $\to$ 266.0).

---

## 5. Root Cause Attribution Analysis (TEST Split)

To diagnose why TEST $p95$ reached 318.0, component-level attribution was performed on the full test population ($N = 1,732,544$):

| Configuration | Candidates | p50 | p90 | p95 | p99 | p99.9 | Max | Fanout $\ge 300$ |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| **A. V2 only** | 76,633,796 | 6.0 | 122.0 | **256.0** | 515.0 | 1,041.0 | 3,707 | 68,264 (3.94%) |
| **B. V2 + EXP-01** | 87,108,588 | 8.0 | 137.0 | **267.0** | 523.0 | 1,058.0 | 3,708 | 73,031 (4.22%) |
| **C. V2 + EXP-02** | 86,284,930 | 8.0 | 137.0 | **284.0** | 573.0 | 1,208.0 | 4,122 | 80,190 (4.63%) |
| **D. V2 + EXP-03** | 90,723,204 | 8.0 | 144.0 | **298.0** | 620.0 | 1,345.0 | 4,193 | 85,848 (4.96%) |
| **E. V2 + EXP-04** | 79,997,593 | 7.0 | 128.0 | **266.0** | 542.0 | 1,070.0 | 3,707 | 72,188 (4.17%) |
| **F. V2 + 02 + 03 + 04** *(No EXP-01)* | 94,254,001 | 8.0 | 150.0 | **307.0** | 646.0 | 1,368.0 | 4,294 | 89,911 (5.19%) |
| **G. Full V3** | 104,705,486 | 10.0 | 163.0 | **318.0** | 653.0 | 1,387.5 | 4,295 | 94,047 (5.43%) |

### Key Findings
1. **EXP-01 is NOT the Root Cause**: Configuration F (completely omitting EXP-01) already reaches $p95 = 307.0 > 300.0$. An experiment tightening EXP-01 cap from 100 to 70 resulted in **0.0 change** to test $p95$.
2. **Interaction Between EXP-02 and EXP-03**: Both rules matched on dense `house_norm` blocks without any per-$S_1$ fanout cap. For the top 10 test entities, EXP-02 and EXP-03 each contributed 1,000–2,800 candidates, inflating the 95th percentile past the top 5% boundary (86,627 entities).

---

## 6. Controlled Cap Sweep & Selection of Experiment D

A controlled ablation sweep tested deterministic block-level caps on EXP-RET-02 and EXP-RET-03:

| Configuration | EXP-02 Cap | EXP-03 Cap | Train Recall | Test p95 | Test p99 | Train Candidates | Test Candidates | Gate Status |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Exp A** (Control) | *None* | *None* | 69.9492% | 277.0 | 555.0 | 84,242,627 | 90,455,055 | PASS BOTH |
| **Exp B** | 100 | *None* | 70.9018% | 289.0 | 559.0 | 89,321,755 | 95,968,409 | PASS BOTH |
| **Exp C** | *None* | 100 | 71.8620% | 290.0 | 559.0 | 95,432,827 | 97,310,724 | PASS BOTH |
| **Exp D** *(Selected)* | **50** | **50** | **72.1619%** | **280.0** | **556.0** | **93,171,949** | **95,196,595** | **PASS BOTH** |
| **Exp E** | 30 | 30 | 71.9895% | 278.0 | 555.0 | 89,829,295 | 93,735,331 | PASS BOTH |
| **Exp F** | 25 | 25 | 71.9246% | 278.0 | 555.0 | 88,940,437 | 93,312,038 | PASS BOTH |
| **Exp G** | 20 | 20 | 71.8386% | 277.0 | 555.0 | 88,002,203 | 92,842,826 | PASS BOTH |

### Selection Justification
**Experiment D** was selected as the **highest-recall configuration among the tested configurations satisfying the Phase 3 gates**.
It achieves **72.1619%** train recall (+6.3995% over V2, preserving 99.64% of uncapped V3 recall) while driving test $p95$ down from 318.0 to **280.0**, comfortably below the 300.0 ceiling.

---

## 7. Final Production V3 Specifications & Verified Metrics

### A. Production V3 Architecture
Production V3 generates candidates symmetrically for Train and Test using:
- **Frozen V2 Baseline Rules** (Rules A through I)
- **EXP-RET-01**: Capped at $\le 100$ per $S_1$ entity
- **EXP-RET-02**: Capped at $\le 50$ per $S_1$ entity (`HAVING count(*) <= 50`)
- **EXP-RET-03**: Capped at $\le 50$ per $S_1$ entity (`HAVING count(*) <= 50`)
- **EXP-RET-04**: Selective locality + token matching

### B. Evaluator Metrics (From Scratch)

#### TRAIN Split ($N = 2,206,821$; Ground Truth: 7,638,365 pairs)
- **$S_2$ Captured Pairs**: 2,677,201 / 3,693,619 (**72.4818%**)
- **$S_3$ Captured Pairs**: 2,834,785 / 3,944,746 (**71.8623%**)
- **Combined Captured Pairs**: **5,511,986 / 7,638,365** (**72.161857%**)
- **Missed True Pairs**: 2,126,379
- **Total Train Candidates**: 93,171,949 ($S_2$: 42,933,945; $S_3$: 50,238,004)
- **Zero-Candidate $S_1$ Count**: 42,938 (1.95%)
- **Train Fanout Quantiles**:
  - $p50$: 9.0
  - $p90$: 123.0
  - **$p95$**: 184.0
  - $p99$: 369.0
  - $p99.9$: 881.0
  - Max Fanout: 3,311

#### TEST Split ($N = 1,732,544$)
- **Total Test Candidates**: 95,196,595 ($S_2$: 43,841,928; $S_3$: 51,354,667)
- **Zero-Candidate $S_1$ Count**: 33,425 (1.93%)
- **Test Fanout Quantiles**:
  - $p50$: 10.0
  - $p90$: 151.0
  - **$p95$**: **280.0** (PASS: $\le 300.0$)
  - $p99$: 556.0
  - $p99.9$: 1,086.0
  - Max Fanout: 3,708
  - Entities with Fanout $\ge 300$: 78,117 (4.509% $\le 5\%$)

#### Candidate Integrity (Both Train & Test)
- Duplicate Pairs: **0**
- Null Entity IDs: **0**
- Malformed Entity IDs: **0**
- Self Matches: **0**
- Cross-Country Leakage: **0**

---

## 8. Authoritative Candidate Artifacts & SHA256 Checksums

The official production V3 candidate files reside in [`P1/data/candidates/v3/`](file:///Users/krishnagera/Amazon-ML-Challenge-2026/P1/data/candidates/v3/):

| File Name | Split | Target | Rows | SHA256 Checksum |
|:---|:---|:---|---:|:---|
| `train_candidate_pairs_s2_v3.tsv` | Train | S2 | 42,933,945 | `61d5dce6389bf424daf61f842ed0f41ebc3991fa27763d4641c4ab4931a7508c` |
| `train_candidate_pairs_s3_v3.tsv` | Train | S3 | 50,238,004 | `d5e90bb6d9a5d81154a83366d98211352e95bebcbc84c5011f14cd0bfe10eeeb` |
| `test_candidate_pairs_s2_v3.tsv` | Test | S2 | 43,841,928 | `4aa0c71ea70a98aff77088da2c4b070e8fb305b8fbebfbda4daa4fa14398dfa2` |
| `test_candidate_pairs_s3_v3.tsv` | Test | S3 | 51,354,667 | `f24041fbbc3a5f56a4b9c26b20dfde3377770f85427a9373bc9c1efc8bf441cd` |

---

## 9. Downstream Pipeline Wiring & P2 Handoff
Downstream scripts in `P2/scripts/` were previously hardcoded to read legacy Strategy B candidates from `outputs/person1_step1/test_candidate_pairs_s*.tsv`.

### Updates Completed
1. [`P2/scripts/phase3_to_8_pipeline.py`](file:///Users/krishnagera/Amazon-ML-Challenge-2026/P2/scripts/phase3_to_8_pipeline.py): Updated `TEST_S2_CAND` and `TEST_S3_CAND` to default to `P1/data/candidates/v3/test_candidate_pairs_s{2,3}_v3.tsv`.
2. [`P2/scripts/finish_pipeline.py`](file:///Users/krishnagera/Amazon-ML-Challenge-2026/P2/scripts/finish_pipeline.py): Updated `TEST_S3_CAND` to default to `P1/data/candidates/v3/test_candidate_pairs_s3_v3.tsv`.
3. [`P2/scripts/regenerate_features.py`](file:///Users/krishnagera/Amazon-ML-Challenge-2026/P2/scripts/regenerate_features.py): Updated test tasks to default to production V3 candidates.
4. **Handoff Document**: Published [`P1/reports/P2_V3_HANDOFF.md`](file:///Users/krishnagera/Amazon-ML-Challenge-2026/P1/reports/P2_V3_HANDOFF.md) instructing Person 2 to consume only validated V3 candidate artifacts.

---

## 10. Known Limitations & Engineering Reality
1. **Candidate Recall Upper Bound**: The achieved combined recall is **72.1619%**. Candidate generation is an upper bound on end-to-end model performance; any true match not captured in this candidate universe (2,126,379 pairs) cannot be scored by downstream ranking models.
2. **Dense Address Ambiguity**: Entities sharing highly common street names or house numbers without distinct brand names remain uncaptured due to fanout caps. Further recall gains would require cross-field semantic embeddings or multi-token street-level blocking.

---

## 11. Phase 3 Acceptance Checklist

| Item | Requirement | Actual Measured | Status |
|:---|:---|:---|:---:|
| 1 | Baseline V2 Recall Exceeded | 72.1619% vs 65.7623% (+6.40% gain) | **PASS** |
| 2 | Test Combined Fanout $p95 \le 300.0$ | **280.0** | **PASS** |
| 3 | Train Combined Fanout $p95 \le 300.0$ | **184.0** | **PASS** |
| 4 | Test Zero-Candidate $S_1$ Entities | 33,425 (1.93%) | **PASS** |
| 5 | Train/Test Retrieval Parity | Symmetrical DuckDB rules across splits | **PASS** |
| 6 | Candidate Integrity Checks | 0 dups, 0 nulls, 0 invalid, 0 self-matches | **PASS** |
| 7 | Zero Cross-Country Leakage | 0 pairs | **PASS** |
| 8 | Reproducibility & Provenance Recorded | Documented in `PHASE3_REPRODUCIBILITY.md` | **PASS** |
| 9 | Downstream Consumers Wired to V3 | `phase3_to_8_pipeline.py`, `finish_pipeline.py` updated | **PASS** |
| 10 | Handoff Documentation Published | Documented in `P2_V3_HANDOFF.md` | **PASS** |
