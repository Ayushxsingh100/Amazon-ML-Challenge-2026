import os
import time
import json
import duckdb
import numpy as np
import lightgbm as lgb

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
NORM_DIR = os.path.join(REPO, "outputs", "person1_step1", "normalized")
FEAT_DIR = os.path.join(REPO, "P2", "data", "features")
PROC_DIR = os.path.join(REPO, "P2", "data", "processed")
MODEL_DIR = os.path.join(REPO, "P2", "models")
REPORT_DIR = os.path.join(REPO, "P2", "reports")
TMP_DIR = os.path.join(REPO, "P2", "data", "duckdb_feat_tmp")

TEST_S3_CAND = os.path.join(REPO, "outputs", "person1_step1", "test_candidate_pairs_s3.tsv")
model_path = os.path.join(MODEL_DIR, "lgbm_baseline_v1.txt")

FEATURE_SQL = """
    -- Name features
    CASE WHEN s1.business_name <> '' AND s1.business_name = tgt.business_name THEN 1.0 ELSE 0.0 END AS name_exact_match,
    jaro_winkler_similarity(COALESCE(s1.business_name,''), COALESCE(tgt.business_name,'')) / 100.0 AS name_jaro_winkler,
    CASE WHEN LENGTH(COALESCE(s1.business_name,'')) >= 2 AND LENGTH(COALESCE(tgt.business_name,'')) >= 2 THEN jaccard(COALESCE(s1.business_name,''), COALESCE(tgt.business_name,'')) ELSE 0.0 END AS name_jaccard,
    CASE WHEN LENGTH(s1.business_name) >= 4 AND LENGTH(tgt.business_name) >= 4
              AND LEFT(s1.business_name, 4) = LEFT(tgt.business_name, 4) THEN 1.0 ELSE 0.0 END AS prefix4_match,
    CASE WHEN s1.business_name <> '' AND tgt.business_name <> ''
              AND SPLIT_PART(s1.business_name, ' ', 1) = SPLIT_PART(tgt.business_name, ' ', 1) THEN 1.0 ELSE 0.0 END AS first_token_match,
    CAST(ABS(LENGTH(COALESCE(s1.business_name,'')) - LENGTH(COALESCE(tgt.business_name,''))) AS DOUBLE) AS name_len_diff,
    CASE WHEN LENGTH(s1.business_name) > 0 AND LENGTH(tgt.business_name) > 0
         THEN CAST(LEAST(LENGTH(s1.business_name), LENGTH(tgt.business_name)) AS DOUBLE) / GREATEST(LENGTH(s1.business_name), LENGTH(tgt.business_name))
         ELSE 0.0 END AS name_len_ratio,
    -- Address features
    CASE WHEN s1.business_address <> '' AND s1.business_address = tgt.business_address THEN 1.0 ELSE 0.0 END AS address_exact_match,
    jaro_winkler_similarity(COALESCE(s1.business_address,''), COALESCE(tgt.business_address,'')) / 100.0 AS address_jaro_winkler,
    CASE WHEN LENGTH(COALESCE(s1.business_address,'')) >= 2 AND LENGTH(COALESCE(tgt.business_address,'')) >= 2 THEN jaccard(COALESCE(s1.business_address,''), COALESCE(tgt.business_address,'')) ELSE 0.0 END AS address_jaccard,
    CAST(ABS(LENGTH(COALESCE(s1.business_address,'')) - LENGTH(COALESCE(tgt.business_address,''))) AS DOUBLE) AS address_len_diff,
    CASE WHEN LENGTH(s1.business_address) > 0 AND LENGTH(tgt.business_address) > 0
         THEN CAST(LEAST(LENGTH(s1.business_address), LENGTH(tgt.business_address)) AS DOUBLE) / GREATEST(LENGTH(s1.business_address), LENGTH(tgt.business_address))
         ELSE 0.0 END AS address_len_ratio,
    -- House number overlap
    CASE WHEN regexp_extract(COALESCE(s1.business_address,''), '\\b[0-9]+\\b', 0) <> ''
              AND regexp_extract(COALESCE(tgt.business_address,''), '\\b[0-9]+\\b', 0) <> ''
              AND regexp_extract(s1.business_address, '\\b[0-9]+\\b', 0) = regexp_extract(tgt.business_address, '\\b[0-9]+\\b', 0)
         THEN 1.0 ELSE 0.0 END AS address_first_number_match,
    -- Country
    CASE WHEN s1.country <> '' AND s1.country = tgt.country THEN 1.0 ELSE 0.0 END AS country_match,
    -- Lengths for model context
    CAST(LENGTH(COALESCE(s1.business_name,'')) AS DOUBLE) AS s1_name_len,
    CAST(LENGTH(COALESCE(tgt.business_name,'')) AS DOUBLE) AS tgt_name_len,
    CAST(LENGTH(COALESCE(s1.business_address,'')) AS DOUBLE) AS s1_addr_len,
    CAST(LENGTH(COALESCE(tgt.business_address,'')) AS DOUBLE) AS tgt_addr_len
"""

FEATURE_NAMES = [
    "name_exact_match", "name_jaro_winkler", "name_jaccard",
    "prefix4_match", "first_token_match", "name_len_diff", "name_len_ratio",
    "address_exact_match", "address_jaro_winkler", "address_jaccard",
    "address_len_diff", "address_len_ratio", "address_first_number_match",
    "country_match",
    "s1_name_len", "tgt_name_len", "s1_addr_len", "tgt_addr_len",
    "source_is_s3",
]

def main():
    print("Loading model...")
    model = lgb.Booster(model_file=model_path)
    
    con = duckdb.connect()
    con.execute("SET memory_limit='4GB'")
    con.execute("SET threads=4")
    con.execute("SET preserve_insertion_order=false")
    con.execute(f"SET temp_directory='{TMP_DIR}'")
    
    for name, path in [
        ("test_s1", os.path.join(NORM_DIR, "test_source1_normalized.tsv")),
        ("test_s3", os.path.join(NORM_DIR, "test_source3_normalized.tsv")),
    ]:
        con.execute(f"""
            CREATE OR REPLACE TABLE {name} AS
            SELECT entity_id, business_name, business_address, country
            FROM read_csv('{path}', delim='\\t', header=true, all_varchar=true)
        """)
        
    test_feat_path = os.path.join(FEAT_DIR, "test_features_s3.tsv")
    out_path = os.path.join(PROC_DIR, "test_pair_predictions_s3.tsv")
    
    if not os.path.exists(test_feat_path):
        print(f"\nS3: Generating test features...")
        con.execute(f"""
            COPY (
                SELECT
                    c.source1_entity_id,
                    c.matched_entity_id,
                    {FEATURE_SQL},
                    1 AS source_is_s3
                FROM read_csv('{TEST_S3_CAND}', delim='\\t', header=true, all_varchar=true) c
                LEFT JOIN test_s1 s1 ON c.source1_entity_id = s1.entity_id
                LEFT JOIN test_s3 tgt ON c.matched_entity_id = tgt.entity_id
            )
            TO '{test_feat_path}'
            WITH (FORMAT CSV, DELIMITER '\\t', HEADER)
        """)
    else:
        print("S3 test features already exist.")
        
    if not os.path.exists(out_path):
        print(f"S3: Scoring...")
        t0 = time.time()
        col_expr = ", ".join([f"CAST({c} AS DOUBLE) AS {c}" for c in FEATURE_NAMES])
        test_data = con.execute(f"""
            SELECT
                source1_entity_id,
                matched_entity_id,
                {col_expr}
            FROM read_csv('{test_feat_path}', delim='\\t', header=true)
        """).fetchnumpy()

        test_s1_ids = test_data["source1_entity_id"]
        test_t_ids = test_data["matched_entity_id"]
        X_test = np.column_stack([test_data[c].astype(np.float32) for c in FEATURE_NAMES])

        scores = model.predict(X_test)

        with open(out_path, "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tcandidate_entity_id\ttarget_source\tmodel_score\n")
            for i in range(len(scores)):
                f.write(f"{test_s1_ids[i]}\t{test_t_ids[i]}\tS3\t{scores[i]:.6f}\n")

        above_50 = (scores >= 0.5).sum()
        above_30 = (scores >= 0.3).sum()
        print(f"    >=0.5: {above_50:,} ({above_50/len(scores)*100:.2f}%)")
        print(f"    >=0.3: {above_30:,} ({above_30/len(scores)*100:.2f}%)")
    else:
        print("S3 already scored.")
        
    print("\nWriting handoff docs...")
    
    experiment = {
        "model_type": "LightGBM",
        "feature_list": FEATURE_NAMES,
        "feature_count": len(FEATURE_NAMES),
        "training_data": {
            "total_pairs": 35591815,
            "positives": 4104011,
            "negatives": 31487804,
            "positive_rate_pct": 11.5308,
        },
        "candidate_strategy": "Strategy B (Rule A: country+prefix4+house UNION Rule B: country+exact_address)",
        "validation": {
            "split": "15% stratified holdout",
            "pr_auc": 0.9982,
        },
        "model_path": model_path,
        "test_output_s2": os.path.join(PROC_DIR, "test_pair_predictions_s2.tsv"),
        "test_output_s3": out_path,
    }

    exp_path = os.path.join(REPORT_DIR, "model_experiment.json")
    with open(exp_path, "w", encoding="utf-8") as f:
        json.dump(experiment, f, indent=2)

    handoff_path = os.path.join(REPORT_DIR, "MODELING_HANDOFF.md")
    with open(handoff_path, "w", encoding="utf-8") as f:
        f.write(f"""# P2 Modeling Handoff

## Model
- **Type**: LightGBM (CPU, 500 rounds)
- **Features**: {len(FEATURE_NAMES)} features
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
""")

    print(f"Handoff doc created: {handoff_path}")
    print("\nPhase 3-8 complete!")

if __name__ == "__main__":
    main()
