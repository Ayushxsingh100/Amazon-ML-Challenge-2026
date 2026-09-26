# Experiment Report: EXP-RET-02/03 Candidate Fanout Pruning Sweep

## Overview
This controlled experiment sweep evaluated the impact of applying per-$S_1$ fanout caps to retrieval components **EXP-RET-02** (Relaxed Name + House) and **EXP-RET-03** (Indic / Transliteration-Aware Retrieval) to solve the Phase 3 TEST combined fanout violation ($p95 = 318.0 > 300.0$) while preserving high train recall.

---

## 1. Summary of Results

| Configuration | EXP-02 Cap | EXP-03 Cap | Train Combined Recall | Test Combined p95 | Test Combined p99 | Train Total Candidates | Test Total Candidates | Train Zero-S1 | Test Zero-S1 | Integrity Status | Gate Status |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **EXPERIMENT A** (Control) | *None* | *None* | 69.9492% | 277.0 | 555.0 | 84,242,627 | 90,455,055 | 54,558 | 39,697 | PASS | PASS BOTH |
| **EXPERIMENT B** | 100 | *None* | 70.9018% | 289.0 | 559.0 | 89,321,755 | 95,968,409 | 47,765 | 33,692 | PASS | PASS BOTH |
| **EXPERIMENT C** | *None* | 100 | 71.8620% | 290.0 | 559.0 | 95,432,827 | 97,310,724 | 44,343 | 33,434 | PASS | PASS BOTH |
| **EXPERIMENT D** *(Selected)* | **50** | **50** | **72.1619%** | **280.0** | **556.0** | **93,171,949** | **95,196,595** | **42,938** | **33,425** | **PASS** | **PASS BOTH** |
| **EXPERIMENT E** | 30 | 30 | 71.9895% | 278.0 | 555.0 | 89,829,295 | 93,735,331 | 43,602 | 33,446 | PASS | PASS BOTH |
| **EXPERIMENT F** | 25 | 25 | 71.9246% | 278.0 | 555.0 | 88,940,437 | 93,312,038 | 43,899 | 33,460 | PASS | PASS BOTH |
| **EXPERIMENT G** | 20 | 20 | 71.8386% | 277.0 | 555.0 | 88,002,203 | 92,842,826 | 44,259 | 33,500 | PASS | PASS BOTH |

---

## 2. Selection of Candidate Configuration

Among tested configurations satisfying both Phase 3 gates (TEST $p95 \le 300.0$ and TRAIN recall $> 65.7623\%$):

### **Highest-Recall Configuration Among Tested Configurations**:
**EXPERIMENT D: EXP-02 cap = 50, EXP-03 cap = 50**

- **TRAIN Combined Recall**: **72.161857%** (5,511,986 captured / 7,638,365 ground truth pairs)
  - V2 Baseline Recall: 65.7623% (+6.3995 percentage points improvement)
  - Uncapped V3 Recall: 72.4229% (preserves 99.64% of V3 recall, sacrificing only -0.2610 percentage points)
  - S2 Recall: 72.4818% (2,677,201 / 3,693,619)
  - S3 Recall: 71.8623% (2,834,785 / 3,944,746)
  - Missed pairs: 2,126,379
- **TEST Combined Fanout**:
  - $p50 = 8.0$
  - $p90 = 148.0$
  - **$p95 = 280.0 \le 300.0$** (PASS, reduced by 38.0 fanout units from 318.0)
  - $p99 = 556.0$
  - $p99.9 = 1,090.0$
  - Max fanout: 3,708
  - S1 entities with fanout $\ge 300$: 78,117 (4.509% of test S1 population $\le 5\%$)
  - S1 entities with zero candidates: 33,425 (1.929%)
- **Integrity**: 0 duplicate pairs, 0 null IDs, 0 self-matches, 0 cross-country matches.

---

## 3. Why EXP-01 CAP70 Was Not Pursued Further
In the root-cause attribution analysis on the test split:
1. Top high-fanout S1 entities (with total fanout between 2,900 and 4,295) receive only **0 to 3 candidates** from EXP-RET-01.
2. EXP-RET-01 was already bounded by an S1 cap of 100, which successfully limited its tail impact.
3. Tightening EXP-01 from cap 100 to cap 70 reduced train recall by 4,300 pairs while leaving test $p95$ completely unchanged at 318.0.
4. Conversely, EXP-RET-02 and EXP-RET-03 each contributed 1,000 to 2,800 candidates to dense address blocks because neither had any fanout cap.
5. Capping EXP-02 and EXP-03 directly targets the true source of tail inflation.

---

## 4. Cap Semantics & Provenance

### Implementation
Neither EXP-RET-02 nor EXP-RET-03 candidate tables retain an intermediate similarity score column on disk. Truncating rows using an arbitrary limit would break determinism and risk discarding valid true pairs.
Therefore, following the established methodology of EXP-RET-01, the cap is implemented as an exact deterministic block-level inclusion threshold:
```sql
SELECT r.source1_entity_id, r.matched_entity_id
FROM exp_raw r
JOIN (
    SELECT source1_entity_id 
    FROM exp_raw 
    GROUP BY source1_entity_id 
    HAVING count(*) <= CAP
) k ON r.source1_entity_id = k.source1_entity_id;
```
If an $S_1$ entity produces more than `CAP` candidates under that retrieval rule, the rule is considered non-selective for that specific entity (e.g. house number '1' matched against thousands of common businesses) and its candidates from that rule are pruned.

### Execution Script
- Script: [`P1/experiments/phase3/EXP-RET-02_03_CAP_SWEEP/run_cap_sweep.py`](file:///Users/krishnagera/Amazon-ML-Challenge-2026/P1/experiments/phase3/EXP-RET-02_03_CAP_SWEEP/run_cap_sweep.py)
- Input paths:
  - Canonical Parquets: `P1/data/entities/{train,test}/source{1,2,3}/`
  - V2 Baselines: `P2/data/candidates/{train,test}_candidate_pairs_{s2,s3}_v2.tsv`
  - Ground truth: `data/train/train_ground_truth.tsv`
  - Component files: `P1/experiments/phase3/EXP-RET-0{1,2,3,4}/`

---

## 5. Artifact SHA256 Checksums (Selected Configuration D)

The generated candidate files for Configuration D are stored under `P1/experiments/phase3/EXP-RET-02_03_CAP_SWEEP/candidates/`:

| Split | Target | File Name | Rows | SHA256 Checksum |
|:---|:---|:---|---:|:---|
| Train | S2 | `train_candidate_pairs_s2_selected.tsv` | 42,933,945 | `61d5dce6389bf424daf61f842ed0f41ebc3991fa27763d4641c4ab4931a7508c` |
| Train | S3 | `train_candidate_pairs_s3_selected.tsv` | 50,238,004 | `d5e90bb6d9a5d81154a83366d98211352e95bebcbc84c5011f14cd0bfe10eeeb` |
| Test | S2 | `test_candidate_pairs_s2_selected.tsv` | 43,841,928 | `4aa0c71ea70a98aff77088da2c4b070e8fb305b8fbebfbda4daa4fa14398dfa2` |
| Test | S3 | `test_candidate_pairs_s3_selected.tsv` | 51,354,667 | `f24041fbbc3a5f56a4b9c26b20dfde3377770f85427a9373bc9c1efc8bf441cd` |

---

## 6. Phase 3 Gate Audit
1. **TEST combined $p95 \le 300.0$**: **PASS** (Measured: **280.0**)
2. **TRAIN combined recall $> 65.7623\%$**: **PASS** (Measured: **72.1619%**)
3. **Deterministic & Reproducible**: **PASS**
4. **Data Integrity (No duplicates, nulls, self-matches, cross-country leakage)**: **PASS**
