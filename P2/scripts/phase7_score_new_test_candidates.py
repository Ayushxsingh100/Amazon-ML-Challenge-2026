#!/usr/bin/env python3
"""
P2/scripts/phase7_score_new_test_candidates.py

Identifies and scores the incremental V4 candidate pairs on the Test split
using the 5-fold LightGBM ensemble.
Outputs:
- P2/predictions/phase7/new_test_predictions_v4.parquet
"""

import time
import duckdb
import numpy as np
import lightgbm as lgb
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

print("=" * 80)
print("PHASE 7: SCORING INCREMENTAL V4 TEST CANDIDATES")
print("=" * 80)

t0 = time.time()
con = duckdb.connect()
con.execute("SET threads=4")
con.execute("SET memory_limit='8GB'")
con.execute("SET preserve_insertion_order=false")

s2_pred = REPO_ROOT / "P2" / "predictions" / "phase4" / "test_predictions_s2.tsv"
s3_pred = REPO_ROOT / "P2" / "predictions" / "phase4" / "test_predictions_s3.tsv"

CANONICAL_FEATURES = [
    'name_exact_match', 'name_jaro_winkler', 'name_jaccard', 'prefix4_match', 'first_token_match',
    'name_len_diff', 'name_len_ratio', 'address_exact_match', 'address_jaro_winkler', 'address_jaccard',
    'address_len_diff', 'address_len_ratio', 'address_first_number_match', 'country_match',
    's1_name_len', 'tgt_name_len', 's1_addr_len', 'tgt_addr_len', 'source_is_s3'
]

FEATURE_SQL = '''
    CASE WHEN s1.business_name <> '' AND s1.business_name = tgt.business_name THEN 1.0 ELSE 0.0 END AS name_exact_match,
    jaro_winkler_similarity(COALESCE(s1.business_name,''), COALESCE(tgt.business_name,'')) / 100.0 AS name_jaro_winkler,
    CASE WHEN LENGTH(COALESCE(s1.business_name,'')) >= 2 AND LENGTH(COALESCE(tgt.business_name,'')) >= 2 THEN jaccard(COALESCE(s1.business_name,''), COALESCE(tgt.business_name,'')) ELSE 0.0 END AS name_jaccard,
    CASE WHEN LENGTH(s1.business_name) >= 4 AND LENGTH(tgt.business_name) >= 4 AND LEFT(s1.business_name, 4) = LEFT(tgt.business_name, 4) THEN 1.0 ELSE 0.0 END AS prefix4_match,
    CASE WHEN s1.business_name <> '' AND tgt.business_name <> '' AND SPLIT_PART(s1.business_name, ' ', 1) = SPLIT_PART(tgt.business_name, ' ', 1) THEN 1.0 ELSE 0.0 END AS first_token_match,
    CAST(ABS(LENGTH(COALESCE(s1.business_name,'')) - LENGTH(COALESCE(tgt.business_name,''))) AS DOUBLE) AS name_len_diff,
    CASE WHEN LENGTH(s1.business_name) > 0 AND LENGTH(tgt.business_name) > 0 THEN CAST(LEAST(LENGTH(s1.business_name), LENGTH(tgt.business_name)) AS DOUBLE) / GREATEST(LENGTH(s1.business_name), LENGTH(tgt.business_name)) ELSE 0.0 END AS name_len_ratio,
    CASE WHEN s1.business_address <> '' AND s1.business_address = tgt.business_address THEN 1.0 ELSE 0.0 END AS address_exact_match,
    jaro_winkler_similarity(COALESCE(s1.business_address,''), COALESCE(tgt.business_address,'')) / 100.0 AS address_jaro_winkler,
    CASE WHEN LENGTH(COALESCE(s1.business_address,'')) >= 2 AND LENGTH(COALESCE(tgt.business_address,'')) >= 2 THEN jaccard(COALESCE(s1.business_address,''), COALESCE(tgt.business_address,'')) ELSE 0.0 END AS address_jaccard,
    CAST(ABS(LENGTH(COALESCE(s1.business_address,'')) - LENGTH(COALESCE(tgt.business_address,''))) AS DOUBLE) AS address_len_diff,
    CASE WHEN LENGTH(s1.business_address) > 0 AND LENGTH(tgt.business_address) > 0 THEN CAST(LEAST(LENGTH(s1.business_address), LENGTH(tgt.business_address)) AS DOUBLE) / GREATEST(LENGTH(s1.business_address), LENGTH(tgt.business_address)) ELSE 0.0 END AS address_len_ratio,
    CASE WHEN regexp_extract(COALESCE(s1.business_address,''), '[0-9]+[A-Za-z]?', 0) <> '' AND regexp_extract(COALESCE(tgt.business_address,''), '[0-9]+[A-Za-z]?', 0) <> '' AND regexp_extract(s1.business_address, '[0-9]+[A-Za-z]?', 0) = regexp_extract(tgt.business_address, '[0-9]+[A-Za-z]?', 0) THEN 1.0 ELSE 0.0 END AS address_first_number_match,
    CASE WHEN s1.country <> '' AND s1.country = tgt.country THEN 1.0 ELSE 0.0 END AS country_match,
    CAST(LENGTH(COALESCE(s1.business_name,'')) AS DOUBLE) AS s1_name_len,
    CAST(LENGTH(COALESCE(tgt.business_name,'')) AS DOUBLE) AS tgt_name_len,
    CAST(LENGTH(COALESCE(s1.business_address,'')) AS DOUBLE) AS s1_addr_len,
    CAST(LENGTH(COALESCE(tgt.business_address,'')) AS DOUBLE) AS tgt_addr_len
'''

print("[1/3] Generating and deduplicating new V4 candidates on Test split...")

con.execute(f"""
CREATE OR REPLACE TABLE new_test_cands_scored AS
WITH test_s1 AS (
    SELECT entity_id, business_name_normalized as business_name, business_address_normalized as business_address, country_normalized as country,
           regexp_replace(business_name_normalized, '\\\\b(inc|llc|ltd|limited|pvt|co|corp|services|company|group)\\\\b', '', 'g') as name_clean,
           left(regexp_replace(lower(coalesce(business_address_normalized,'')), '[^a-z0-9]', '', 'g'), 20) as addr_clean20
    FROM read_parquet('P1/data/entities/test/source1/test_s1_entities.parquet')
),
test_s2 AS (
    SELECT entity_id, business_name_normalized as business_name, business_address_normalized as business_address, country_normalized as country,
           regexp_replace(business_name_normalized, '\\\\b(inc|llc|ltd|limited|pvt|co|corp|services|company|group)\\\\b', '', 'g') as name_clean,
           left(regexp_replace(lower(coalesce(business_address_normalized,'')), '[^a-z0-9]', '', 'g'), 20) as addr_clean20
    FROM read_parquet('P1/data/entities/test/source2/test_s2_entities.parquet')
),
test_s3 AS (
    SELECT entity_id, business_name_normalized as business_name, business_address_normalized as business_address, country_normalized as country,
           regexp_replace(business_name_normalized, '\\\\b(inc|llc|ltd|limited|pvt|co|corp|services|company|group)\\\\b', '', 'g') as name_clean,
           left(regexp_replace(lower(coalesce(business_address_normalized,'')), '[^a-z0-9]', '', 'g'), 20) as addr_clean20
    FROM read_parquet('P1/data/entities/test/source3/test_s3_entities.parquet')
),

-- Strategy A & B on S2
strat_a_s2_raw AS (
    SELECT s1.entity_id as source1_entity_id, tgt.entity_id as candidate_entity_id
    FROM test_s1 s1 JOIN test_s2 tgt ON s1.country = tgt.country AND left(s1.addr_clean20, 15) = left(tgt.addr_clean20, 15)
    WHERE length(left(s1.addr_clean20, 15)) >= 12
),
strat_a_s2 AS (
    SELECT r.source1_entity_id, r.candidate_entity_id FROM strat_a_s2_raw r
    JOIN (SELECT source1_entity_id FROM strat_a_s2_raw GROUP BY 1 HAVING count(*) <= 30) k USING (source1_entity_id)
),
strat_b_s2_raw AS (
    SELECT s1.entity_id as source1_entity_id, tgt.entity_id as candidate_entity_id
    FROM test_s1 s1 JOIN test_s2 tgt ON s1.country = tgt.country AND regexp_replace(lower(s1.name_clean), '[^a-z0-9]', '', 'g') = regexp_replace(lower(tgt.name_clean), '[^a-z0-9]', '', 'g')
    WHERE length(regexp_replace(lower(s1.name_clean), '[^a-z0-9]', '', 'g')) >= 8
),
strat_b_s2 AS (
    SELECT r.source1_entity_id, r.candidate_entity_id FROM strat_b_s2_raw r
    JOIN (SELECT source1_entity_id FROM strat_b_s2_raw GROUP BY 1 HAVING count(*) <= 30) k USING (source1_entity_id)
),

-- Strategy A & B on S3
strat_a_s3_raw AS (
    SELECT s1.entity_id as source1_entity_id, tgt.entity_id as candidate_entity_id
    FROM test_s1 s1 JOIN test_s3 tgt ON s1.country = tgt.country AND left(s1.addr_clean20, 15) = left(tgt.addr_clean20, 15)
    WHERE length(left(s1.addr_clean20, 15)) >= 12
),
strat_a_s3 AS (
    SELECT r.source1_entity_id, r.candidate_entity_id FROM strat_a_s3_raw r
    JOIN (SELECT source1_entity_id FROM strat_a_s3_raw GROUP BY 1 HAVING count(*) <= 30) k USING (source1_entity_id)
),
strat_b_s3_raw AS (
    SELECT s1.entity_id as source1_entity_id, tgt.entity_id as candidate_entity_id
    FROM test_s1 s1 JOIN test_s3 tgt ON s1.country = tgt.country AND regexp_replace(lower(s1.name_clean), '[^a-z0-9]', '', 'g') = regexp_replace(lower(tgt.name_clean), '[^a-z0-9]', '', 'g')
    WHERE length(regexp_replace(lower(s1.name_clean), '[^a-z0-9]', '', 'g')) >= 8
),
strat_b_s3 AS (
    SELECT r.source1_entity_id, r.candidate_entity_id FROM strat_b_s3_raw r
    JOIN (SELECT source1_entity_id FROM strat_b_s3_raw GROUP BY 1 HAVING count(*) <= 30) k USING (source1_entity_id)
),

-- Deduplicate against existing V3 test predictions
new_s2 AS (
    SELECT DISTINCT a.source1_entity_id, a.candidate_entity_id
    FROM (SELECT * FROM strat_a_s2 UNION ALL SELECT * FROM strat_b_s2) a
    LEFT JOIN read_csv('{s2_pred}', delim='\\t', header=true) v 
      ON a.source1_entity_id = v.source1_entity_id AND a.candidate_entity_id = v.candidate_entity_id
    WHERE v.candidate_entity_id IS NULL
),
new_s3 AS (
    SELECT DISTINCT a.source1_entity_id, a.candidate_entity_id
    FROM (SELECT * FROM strat_a_s3 UNION ALL SELECT * FROM strat_b_s3) a
    LEFT JOIN read_csv('{s3_pred}', delim='\\t', header=true) v 
      ON a.source1_entity_id = v.source1_entity_id AND a.candidate_entity_id = v.candidate_entity_id
    WHERE v.candidate_entity_id IS NULL
)

SELECT c.source1_entity_id, c.candidate_entity_id, 0 AS source_is_s3, {FEATURE_SQL}
FROM new_s2 c
JOIN test_s1 s1 ON c.source1_entity_id = s1.entity_id
JOIN test_s2 tgt ON c.candidate_entity_id = tgt.entity_id
UNION ALL
SELECT c.source1_entity_id, c.candidate_entity_id, 1 AS source_is_s3, {FEATURE_SQL}
FROM new_s3 c
JOIN test_s1 s1 ON c.source1_entity_id = s1.entity_id
JOIN test_s3 tgt ON c.candidate_entity_id = tgt.entity_id;
""")

print("[2/3] Extracting features into numpy array...")
data = con.execute("SELECT source1_entity_id, candidate_entity_id, " + ", ".join(CANONICAL_FEATURES) + " FROM new_test_cands_scored").fetchnumpy()
con.execute("DROP TABLE new_test_cands_scored;")

s1_arr = data['source1_entity_id']
cand_arr = data['candidate_entity_id']
X_new = np.column_stack([data[col].astype(np.float32) for col in CANONICAL_FEATURES])
total_new = len(s1_arr)
del data

print(f"Total incremental candidates to score: {total_new:,}")

print("[3/3] Scoring with 5-fold LightGBM ensemble...")
ensemble_scores = np.zeros(total_new, dtype=np.float32)
for fold in range(5):
    bpath = REPO_ROOT / f"P2/models/phase4/lgb_fold{fold}.txt"
    booster = lgb.Booster(model_file=str(bpath))
    fold_preds = booster.predict(X_new)
    ensemble_scores += (fold_preds / 5.0).astype(np.float32)
    print(f"  Scored with fold {fold} booster.")

del X_new

out_parquet = REPO_ROOT / "P2" / "predictions" / "phase7" / "new_test_predictions_v4.parquet"
con.execute("CREATE OR REPLACE TABLE scored_new (source1_entity_id VARCHAR, candidate_entity_id VARCHAR, model_score FLOAT);")
import pyarrow as pa
import pyarrow.parquet as pq

table = pa.Table.from_arrays(
    [pa.array(s1_arr), pa.array(cand_arr), pa.array(ensemble_scores)],
    names=['source1_entity_id', 'candidate_entity_id', 'model_score']
)
pq.write_table(table, str(out_parquet), compression='zstd')
print(f"Successfully saved scored incremental candidates to: {out_parquet}")
print(f"Total runtime: {time.time()-t0:.2f}s")
