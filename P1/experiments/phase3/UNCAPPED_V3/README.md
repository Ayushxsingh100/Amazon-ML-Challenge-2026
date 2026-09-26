# Historical Uncapped V3 Candidate Artifacts (Baseline)

These files represent the initial uncapped V3 baseline before the Phase 3 fanout cap was applied:
- `train_candidate_pairs_s2_v3.tsv`
- `train_candidate_pairs_s3_v3.tsv`
- `test_candidate_pairs_s2_v3.tsv`
- `test_candidate_pairs_s3_v3.tsv`

## Performance
- TRAIN Combined Recall: 72.4229% (5,531,929 captured pairs)
- TRAIN Combined p95: 199.0
- TEST Combined p95: 318.0 (FAILED Phase 3 gate <= 300)
- TEST Total Candidates: 104,705,486

These files are preserved for historical audit and ablation comparison.
Production V3 replaces them with the validated capped artifacts (EXP-02 cap=50, EXP-03 cap=50).
