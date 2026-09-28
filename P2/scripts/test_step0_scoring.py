import os
import sys
import time
import tempfile
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd

repo_root = Path("c:/NEW AMAZON")
s1_test_path = repo_root / "Amazon-ML-Challenge-2026/outputs/person1_step1/normalized/test_source1_normalized.tsv"
s2_test_path = repo_root / "Amazon-ML-Challenge-2026/outputs/person1_step1/normalized/test_source2_normalized.tsv"
s3_test_path = repo_root / "Amazon-ML-Challenge-2026/outputs/person1_step1/normalized/test_source3_normalized.tsv"
model_dir = repo_root / "Amazon-ML-Challenge-2026/P2/models/phase4"

print("--- Step 0: Testing Feature + Predict Pipeline on Arbitrary Candidate Pairs ---")
t0 = time.time()

# Load 5 fold models
def load_lf_model(p: Path) -> lgb.Booster:
    with open(p, "r", encoding="utf-8") as f:
        text = f.read().replace("\r\n", "\n")
    with tempfile.NamedTemporaryFile("w", delete=False, newline="\n", suffix=".txt") as tmp:
        tmp.write(text)
        tmp_path = tmp.name
    try:
        bst = lgb.Booster(model_file=tmp_path)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
    return bst

models = [load_lf_model(model_dir / f"lgb_fold{fold}.txt") for fold in range(5)]
print(f"Loaded 5 LightGBM models in {time.time() - t0:.2f}s")

# Let's test on 10 arbitrary test pairs
con = duckdb.connect()
con.execute("PRAGMA threads=4;")

test_pairs_df = pd.DataFrame([
    {"source1_entity_id": "S1-100005079", "candidate_entity_id": "S2-71223285"},
    {"source1_entity_id": "S1-100005079", "candidate_entity_id": "S3-167390486"},
    {"source1_entity_id": "S1-100024168", "candidate_entity_id": "S2-118140189"}
])

con.register("test_pairs", test_pairs_df)

con.execute(f"""
CREATE TEMP TABLE s1 AS 
SELECT entity_id, business_name, business_address, country 
FROM read_csv('{str(s1_test_path).replace(os.sep, "/")}', delim='\t', header=True, all_varchar=True);

CREATE TEMP TABLE tgt AS 
SELECT entity_id, business_name, business_address, country 
FROM read_csv('{str(s2_test_path).replace(os.sep, "/")}', delim='\t', header=True, all_varchar=True)
UNION ALL
SELECT entity_id, business_name, business_address, country 
FROM read_csv('{str(s3_test_path).replace(os.sep, "/")}', delim='\t', header=True, all_varchar=True);
""")

FEATURE_SQL = """
    p.source1_entity_id,
    p.candidate_entity_id,
    CASE WHEN s1.business_name <> '' AND s1.business_name = tgt.business_name THEN 1.0 ELSE 0.0 END AS name_exact_match,
    jaro_winkler_similarity(COALESCE(s1.business_name,''), COALESCE(tgt.business_name,'')) / 100.0 AS name_jaro_winkler,
    CASE WHEN LENGTH(COALESCE(s1.business_name,'')) >= 2 AND LENGTH(COALESCE(tgt.business_name,'')) >= 2 
         THEN jaccard(COALESCE(s1.business_name,''), COALESCE(tgt.business_name,'')) ELSE 0.0 END AS name_jaccard,
    CASE WHEN LENGTH(s1.business_name) >= 4 AND LENGTH(tgt.business_name) >= 4
              AND LEFT(s1.business_name, 4) = LEFT(tgt.business_name, 4) THEN 1.0 ELSE 0.0 END AS prefix4_match,
    CASE WHEN s1.business_name <> '' AND tgt.business_name <> ''
              AND SPLIT_PART(s1.business_name, ' ', 1) = SPLIT_PART(tgt.business_name, ' ', 1) THEN 1.0 ELSE 0.0 END AS first_token_match,
    CAST(ABS(LENGTH(COALESCE(s1.business_name,'')) - LENGTH(COALESCE(tgt.business_name,''))) AS DOUBLE) AS name_len_diff,
    CASE WHEN LENGTH(s1.business_name) > 0 AND LENGTH(tgt.business_name) > 0
         THEN CAST(LEAST(LENGTH(s1.business_name), LENGTH(tgt.business_name)) AS DOUBLE) / GREATEST(LENGTH(s1.business_name), LENGTH(tgt.business_name))
         ELSE 0.0 END AS name_len_ratio,
    CASE WHEN s1.business_address <> '' AND s1.business_address = tgt.business_address THEN 1.0 ELSE 0.0 END AS address_exact_match,
    jaro_winkler_similarity(COALESCE(s1.business_address,''), COALESCE(tgt.business_address,'')) / 100.0 AS address_jaro_winkler,
    CASE WHEN LENGTH(COALESCE(s1.business_address,'')) >= 2 AND LENGTH(COALESCE(tgt.business_address,'')) >= 2 
         THEN jaccard(COALESCE(s1.business_address,''), COALESCE(tgt.business_address,'')) ELSE 0.0 END AS address_jaccard,
    CAST(ABS(LENGTH(COALESCE(s1.business_address,'')) - LENGTH(COALESCE(tgt.business_address,''))) AS DOUBLE) AS address_len_diff,
    CASE WHEN LENGTH(s1.business_address) > 0 AND LENGTH(tgt.business_address) > 0
         THEN CAST(LEAST(LENGTH(s1.business_address), LENGTH(tgt.business_address)) AS DOUBLE) / GREATEST(LENGTH(s1.business_address), LENGTH(tgt.business_address))
         ELSE 0.0 END AS address_len_ratio,
    CASE WHEN regexp_extract(COALESCE(s1.business_address,''), '[0-9]+[A-Za-z]?', 0) <> ''
              AND regexp_extract(COALESCE(tgt.business_address,''), '[0-9]+[A-Za-z]?', 0) <> ''
              AND regexp_extract(s1.business_address, '[0-9]+[A-Za-z]?', 0) = regexp_extract(tgt.business_address, '[0-9]+[A-Za-z]?', 0)
         THEN 1.0 ELSE 0.0 END AS address_first_number_match,
    CASE WHEN s1.country <> '' AND s1.country = tgt.country THEN 1.0 ELSE 0.0 END AS country_match,
    CAST(LENGTH(COALESCE(s1.business_name,'')) AS DOUBLE) AS s1_name_len,
    CAST(LENGTH(COALESCE(tgt.business_name,'')) AS DOUBLE) AS tgt_name_len,
    CAST(LENGTH(COALESCE(s1.business_address,'')) AS DOUBLE) AS s1_addr_len,
    CAST(LENGTH(COALESCE(tgt.business_address,'')) AS DOUBLE) AS tgt_addr_len,
    CASE WHEN p.candidate_entity_id LIKE 'S3-%' THEN 1.0 ELSE 0.0 END AS source_is_s3
"""

df_features = con.execute(f"""
SELECT {FEATURE_SQL}
FROM test_pairs p
JOIN s1 ON p.source1_entity_id = s1.entity_id
JOIN tgt ON p.candidate_entity_id = tgt.entity_id;
""").df()

print("Computed features shape:", df_features.shape)

feature_cols = [
    "name_exact_match", "name_jaro_winkler", "name_jaccard", "prefix4_match",
    "first_token_match", "name_len_diff", "name_len_ratio",
    "address_exact_match", "address_jaro_winkler", "address_jaccard",
    "address_len_diff", "address_len_ratio", "address_first_number_match",
    "country_match", "s1_name_len", "tgt_name_len", "s1_addr_len",
    "tgt_addr_len", "source_is_s3"
]

X = df_features[feature_cols].values
preds = np.zeros(len(X), dtype=np.float64)
for bst in models:
    preds += bst.predict(X) / 5.0

df_features["score"] = np.round(preds, 4)
print(df_features[["source1_entity_id", "candidate_entity_id", "score"]])
print("Step 0 confirmation successful!")
