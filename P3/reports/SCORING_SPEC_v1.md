# SCORING_SPEC_v1

## Status
FROZEN FOR E02

## Official Scoring Unit
One Source-1 entity.

For each S1:
- Ground truth = set of matched S2/S3 IDs
- Prediction = set of predicted S2/S3 IDs

F0.5 is calculated independently for each S1 and then
macro-averaged across the S1 evaluation population.

## Formula

F0.5 = (1.25 * Precision * Recall) /
       (0.25 * Precision + Recall)

## S2 / S3 Handling

S2 and S3 are scored jointly per S1.

Example:
S1-X -> {S2-A, S3-B, S3-C}

This is one entity-level scoring unit.

S2-only and S3-only scores may be used as diagnostics,
but they are not the canonical E02 score.

## No-Match Handling

True no-match + empty prediction:
F0.5 = 1.0

True no-match + non-empty prediction:
F0.5 = 0.0

## Missed Match

True match(s) + empty prediction:
F0.5 = 0.0

## Partial Match

TP, FP and FN are calculated from set intersection and
set differences at the S1 level.

## Zero-Candidate S1

A zero-candidate S1 has no available candidate pair.

If predicted empty:
- true no-match -> 1.0
- true match exists -> 0.0

No special replacement metric is used.

## Candidate-Generation Misses

If a true match is absent from the candidate set, the
model cannot predict that candidate. The resulting missed
true link contributes to FN in the official S1-level score.

Candidate recall is a separate diagnostic and does not
replace Macro F0.5.

## Determinism

The scorer is deterministic and does not:
- optimize folds,
- tune thresholds,
- inspect E02 results,
- modify candidates,
- modify features.

## Version

scorer_v1

Implementation:
validation/scorer_v1.py

Underlying metric implementation:
validation/metrics.py
