# P2 Modeling Handoff

> [!WARNING]
> **E01 Invalidation Notice**
> E01 pair-level validation is superseded. E01 model/features were trained/scored against the non-canonical old candidate universe (where the regex had \b boundaries). E01 is retained only as an engineering baseline and must NOT be used as competition evidence.
> 
> **cands_B_v1 Invalidation Notice**
> cands_B_v1 is superseded for E02 because its TRAIN/TEST universes were not identical. cands_BCD_v1 is the new canonical E02 candidate universe.

## Model
- **Type**: LightGBM (CPU, 500 rounds)
- **Features**: 19 features
- **Training pairs**: 35,591,815 (pos=4,104,011, neg=31,487,804, rate=11.53%)
- **Candidate Strategy**: Strategy B (Rule A: country+prefix4+house UNION Rule B: country+exact_address)

## Validation Metrics (15% stratified holdout)
- **PR-AUC**: 0.9982

### Threshold Analysis
| Threshold | Precision | Recall | F1 | F0.5 |
|-----------|-----------|--------|-----|------|
| 0.05 | 0.9044 | 0.9981 | 0.9490 | 0.9217 |
| 0.10 | 0.9327 | 0.9971 | 0.9638 | 0.9449 |
| 0.20 | 0.9533 | 0.9954 | 0.9739 | 0.9614 |
| 0.30 | 0.9631 | 0.9940 | 0.9783 | 0.9691 |
| 0.40 | 0.9694 | 0.9925 | 0.9808 | 0.9739 |
| 0.50 | 0.9740 | 0.9909 | 0.9824 | 0.9773 |
| 0.60 | 0.9781 | 0.9889 | 0.9835 | 0.9803 |
| 0.70 | 0.9821 | 0.9861 | 0.9841 | 0.9829 |
| 0.80 | 0.9861 | 0.9812 | 0.9836 | 0.9851 |
| 0.90 | 0.9907 | 0.9702 | 0.9803 | 0.9865 |

## Output Files
| File | Description |
|------|-------------|
| `P2/models/lgbm_baseline_v1.txt` | Trained LightGBM model |
| `P2/data/processed/test_pair_predictions_s2.tsv` | Test S2 scored predictions |
| `P2/data/processed/test_pair_predictions_s3.tsv` | Test S3 scored predictions |
| `P2/reports/error_analysis.tsv` | Validation error analysis |
| `P2/reports/model_experiment.json` | Full experiment record |
| `P2/data/features/train_features_combined.tsv` | Combined training features |

## Notes
- To prevent OOM during model training, negative examples were randomly downsampled to ~33% during memory loading. Positives were kept at 100%. The full candidate TSV (`train_features_combined.tsv`) still contains all 35.5M rows.
- No final production threshold declared. Evidence provided for Person 3 to decide.
- No test labels available; test outputs contain scores only.
- Final `matching_results.tsv` construction is owned by Person 3.
- Blocking recall at Strategy B level: S2=54.22%, S3=53.27%. Model can only score pairs in the candidate set.
