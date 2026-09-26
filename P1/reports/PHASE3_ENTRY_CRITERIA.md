# Phase 3 Entry Criteria Audit

**Auditor**: Person 1 (P1)  
**Date**: September 26, 2026  
**Status**: **PASS (with documented pipeline constraints)**

This document evaluates the 10 mandatory gatekeeping criteria required before commencing Phase 3 retrieval experiments.

---

## Evaluation Gate Matrix

| # | Question | Status | Forensic Evidence |
|---|---|---|---|
| 1 | **Do we have a trusted V1 baseline?** | **PASS** | Evaluated on real GT. Train S2 Recall = 59.9449% (2,214,137 / 3,693,619); Train S3 Recall = 59.4827% (2,346,442 / 3,944,746); Combined = 59.7062% (4,560,579 / 7,638,365). Full manifest and rules verified in [cands_BCD_v1_tasks.py](file:///Users/krishnagera/Amazon-ML-Challenge-2026/P2/scripts/cands_BCD_v1_tasks.py). |
| 2 | **Do we have a trusted V2 baseline?** | **PASS** | Evaluated on real GT. Train S2 Recall = 66.0213% (2,438,575 / 3,693,619); Train S3 Recall = 65.5199% (2,584,593 / 3,944,746); Combined = 65.7623% (5,023,168 / 7,638,365). Full manifest and rules verified in [cands_BCD_v2_tasks.py](file:///Users/krishnagera/Amazon-ML-Challenge-2026/P2/scripts/cands_BCD_v2_tasks.py). |
| 3 | **Can we reproduce their recall?** | **PASS** | 100% bitwise numerical agreement between newly executed evaluations and Phase 0 baseline cache. |
| 4 | **Is the evaluator independently validated?** | **PASS** | [evaluate_candidate_set.py](file:///Users/krishnagera/Amazon-ML-Challenge-2026/P1/scripts/evaluation/evaluate_candidate_set.py) independently verified against all 10 candidate files, matching verified GT counts, entity sets, and fanout statistics. |
| 5 | **Do train and test retrieval rules match?** | **PARTIAL** | [P2/scripts/cands_BCD_v1_tasks.py](file:///Users/krishnagera/Amazon-ML-Challenge-2026/P2/scripts/cands_BCD_v1_tasks.py) and [P2/scripts/cands_BCD_v2_tasks.py](file:///Users/krishnagera/Amazon-ML-Challenge-2026/P2/scripts/cands_BCD_v2_tasks.py) generate test files with identical rules to train (**PASS**). However, legacy scoring scripts ([finish_pipeline.py](file:///Users/krishnagera/Amazon-ML-Challenge-2026/P2/scripts/finish_pipeline.py), [phase3_to_8_pipeline.py](file:///Users/krishnagera/Amazon-ML-Challenge-2026/P2/scripts/phase3_to_8_pipeline.py)) wired to `outputs/person1_step1/` consumed Strategy B candidates only (**FAIL**). |
| 6 | **Are candidate schemas standardized?** | **PASS** | All train files follow `source1_entity_id\tmatched_entity_id\tlabel`. All test files follow `source1_entity_id\tmatched_entity_id`. All 10 files pass schema and integrity checks. |
| 7 | **Are duplicate/invalid/cross-country issues understood?** | **PASS** | All candidate files contain 0 duplicates, 0 invalid entity IDs, 0 nulls, 0 self-matches, and 0 cross-country candidate pairs. |
| 8 | **Do we have candidate fanout statistics?** | **PASS** | Full distribution percentiles (min, median, p90, p95, p99, p99.9, max) and zero-candidate S1 counts documented for all candidate sets in [PHASE2_CANDIDATE_BASELINE.tsv](file:///Users/krishnagera/Amazon-ML-Challenge-2026/P1/reports/PHASE2_CANDIDATE_BASELINE.tsv). |
| 9 | **Can a future retrieval experiment be evaluated with exactly the same methodology?** | **PASS** | Standardized CLI [evaluate_candidate_set.py](file:///Users/krishnagera/Amazon-ML-Challenge-2026/P1/scripts/evaluation/evaluate_candidate_set.py) supports any TSV/Parquet candidate file and outputs exact recall, missed pairs, true candidate rates, and fanout metrics. |
| 10 | **What issues must Phase 3 resolve before V3 is considered valid?** | **PASS** | Detailed checklist below specifies exactly what Phase 3 must resolve before V3 can be accepted. |

---

## Mandatory Phase 3 Resolution Checklist for V3

Before any V3 candidate generation run can be declared valid in Phase 3, the following requirements must be met:

1. **Strict Input Source**: Candidate generation must read from the standardized canonical P1 entities ([P1/data/entities/](file:///Users/krishnagera/Amazon-ML-Challenge-2026/P1/data/entities/)), not ad-hoc raw or legacy normalized files.
2. **Evaluator Execution**: All V3 candidate sets must be evaluated using [evaluate_candidate_set.py](file:///Users/krishnagera/Amazon-ML-Challenge-2026/P1/scripts/evaluation/evaluate_candidate_set.py).
3. **Recall Improvement vs Baseline**: V3 must demonstrate a measurable recall increase over V2 Combined Recall (**65.7623%**) against the real training ground truth.
4. **Fanout Upper Bound**: Candidate explosion must be controlled. The 95th percentile fanout should not exceed 300 candidates per S1 entity, and max fanout must be explicitly monitored.
5. **Zero-Candidate S1 Reduction**: Candidate retrieval must specifically target reducing the 94,043 S1 entities (4.26%) that currently have zero candidates in V2 Combined.
6. **Train / Test Symmetrical Output**: Any new rule introduced for V3 must produce both train and test candidate files simultaneously, using identical SQL/Python logic.
7. **Downstream Pipeline Wiring**: Verification that test prediction and scoring pipelines read from the newly generated V3 test candidate artifacts, completely deprecating references to `outputs/person1_step1/`.
8. **No Model Training in Retrieval Experiments**: Candidate recall expansion must be validated purely as candidate pairs before passing to Person 2 (P2) for feature generation and LightGBM classification.
