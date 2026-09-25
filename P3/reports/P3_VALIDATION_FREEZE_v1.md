P3 VALIDATION FREEZE v1

STATUS
FROZEN FOR E02

CANONICAL MODELING COMMIT
9dfb66980c169f215abc20f3e85bfedcb5908be6

CURRENT GENERATION COMMIT
e18e47e4337e495d3b55468e7fa7c912a4ed42c5

FOLDS
Artifact: P3/reports/folds_v1_manifest.tsv
Total S1 entities: 2206821
Fold count: 5
Seed: 314159
Fold 0: 440272
Fold 1: 442332
Fold 2: 441693
Fold 3: 440942
Fold 4: 441582
SHA256: 9dcec5d83a477d224067e71b93abc21af8befa26f9c399568121fa83ba8801a3

FOLD SPEC
Artifact: P3/reports/folds_v1_spec.md
SHA256: 76C0F5D86AF3481E72887D76CEFD8286D296647E863124183...

SCORER
Version: scorer_v1
Implementation: validation/scorer_v1.py
Underlying implementation: validation/metrics.py
Hand tests: PASSED

SCORER SPEC
Artifact: P3/reports/SCORING_SPEC_v1.md
SHA256: 9DCDE5398AF50E4A103FF3A7B821676DA21491085158186DC...

SCORER TESTS
Artifact: validation/test_scorer_v1.py
SHA256: 0D8DF810976B77272F0E803EEF06701AE8416AECA49DE82FB...

SCORING PROTOCOL
- Unit: Source-1 entity
- Metric: Macro F0.5
- S2 and S3: jointly scored per S1
- True no-match + empty prediction: F0.5 = 1.0
- True no-match + non-empty prediction: F0.5 = 0.0
- True match + empty prediction: F0.5 = 0.0
- Zero-candidate S1: scored using the same official entity-level rule
- Candidate-generation misses contribute to FN
- No threshold tuning in scorer
- No model results used to construct folds

CANONICAL DATA
Candidate family: cands_B_v1
Documented strategy: Hybrid B+C+D

P3 FREEZE DECISION
folds_v1 and scorer_v1 are frozen for E02.
