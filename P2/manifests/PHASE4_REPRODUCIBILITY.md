# Phase 4 Reproducibility Manifest

**Project**: Amazon ML Challenge 2026 — Entity Resolution  
**Phase**: Phase 4 Reproducibility  
**Date**: September 26, 2026  
**Python Version**: `3.13.5`  
**DuckDB Version**: `1.5.5`  
**LightGBM Version**: `4.7.0`  
**Pandas Version**: `3.0.5`  
**NumPy Version**: `2.5.2`  

---

## 1. Input Artifacts & Verified Checksums

```text
61d5dce6389bf424daf61f842ed0f41ebc3991fa27763d4641c4ab4931a7508c  P1/data/candidates/v3/train_candidate_pairs_s2_v3.tsv (42,933,945 rows)
d5e90bb6d9a5d81154a83366d98211352e95bebcbc84c5011f14cd0bfe10eeeb  P1/data/candidates/v3/train_candidate_pairs_s3_v3.tsv (50,238,004 rows)
4aa0c71ea70a98aff77088da2c4b070e8fb305b8fbebfbda4daa4fa14398dfa2  P1/data/candidates/v3/test_candidate_pairs_s2_v3.tsv (43,841,928 rows)
f24041fbbc3a5f56a4b9c26b20dfde3377770f85427a9373bc9c1efc8bf441cd  P1/data/candidates/v3/test_candidate_pairs_s3_v3.tsv (51,354,667 rows)
70bc1d8a16c667e0155c2105d0ab2ebe41d7e7a85d8a529e3ca81c6c3a5af037  outputs/person1_step1/train_ground_truth_reconstructed.tsv (2,206,821 rows)
9dcec5d83ae99ea9254d32e92c42ce2c2ce0f9a23fc26c7104d49a7fa347895e  P3/reports/folds_v1_manifest.tsv (2,206,821 rows)
```

---

## 2. Replication Command

To replicate the entire Phase 4 training, evaluation, threshold sweep, and test prediction generation:

```bash
python3 P2/scripts/phase4_modeling_pipeline.py
```

---

## 3. Model Configuration & Seeds

- **Algorithm**: LightGBM Binary Classifier
- **CV Strategy**: S1-grouped 5-fold CV via `P3/reports/folds_v1_manifest.tsv`
- **Random Seed**: 2026
- **Boosting Rounds**: 500
- **Learning Rate**: 0.05
- **Num Leaves**: 31
- **Min Data In Leaf**: 20
- **Negative Downsampling**: 10% deterministic hash (`ABS(hash(s1 || tgt)) % 10 = 0`) + 100% positives

---

## 4. Output Artifacts & Checksums

```text
81c9c55bf3980cf7af35eff4eaa24bf15ee4d98dd3e4028a92d3ceca51e1cc41  P2/models/phase4/lgb_fold0.txt
24609eac326a62ccad64a997085a94fcf09cbd7e5177834866efbb1ce983af6e  P2/models/phase4/lgb_fold1.txt
1caae0a1a755774a360c8a4d506a5bee56c5a62455db7e58ae484185871b5de6  P2/models/phase4/lgb_fold2.txt
6414e917b0f6e74d38d01558712c1070f416e5205bc8df0dc41c92d423221665  P2/models/phase4/lgb_fold3.txt
6ded4f9508e0d536e1ebd65312153cf2a106d870b4e287950d10da6063d9592b  P2/models/phase4/lgb_fold4.txt
1a4e94cc9b15ba984588db2b763e6c15967b2d7a7ef04213833d79dfe11307af  P2/predictions/phase4/test_predictions_s2.tsv (43,841,928 rows)
503282181cff0df65b86f3f19c2f2813b4cd84d99a1a2b4ed167ac2199d5e3aa  P2/predictions/phase4/test_predictions_s3.tsv (51,354,667 rows)
4a2f527e0a22e89012d864b122d63814ed0e4dd5823b1c2341804ecbff0e8682  P2/reports/phase4_modeling_metrics.json
```
