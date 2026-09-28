# folds_v1 Specification

## Status
FROZEN FOR E02

## Population
- Total S1 entities: 2,206,821
- Key: source1_entity_id
- Every S1 entity appears exactly once.

## Fold Structure
- Number of folds: 5
- Fold IDs: 0, 1, 2, 3, 4
- Split unit: Source-1 entity
- No pair-level random splitting.
- All candidate pairs for one S1 remain in the same fold.
- Zero-candidate and no-match S1 entities are included.

## Deterministic Assignment
Seed: 314159

For each source1_entity_id:

SHA256("314159|" + source1_entity_id)

The first 8 digest bytes are interpreted as an unsigned
big-endian integer and reduced modulo 5.

## Rationale
Validation is performed at S1 entity level, therefore the
validation split is also performed at S1 entity level.

No fold assignment is optimized using ground truth, model
results, candidate recall, features, thresholds, or E02 results.

## Candidate / Feature Freeze
Candidate generation and feature definitions are unchanged.

Canonical candidate family:
cands_B_v1

Documented strategy:
Hybrid B+C+D

## Reproducibility
The generation artifact records:
- Git SHA
- Source-1 SHA256
- seed
- fold count
- entity count
- fold counts
- folds artifact SHA256
