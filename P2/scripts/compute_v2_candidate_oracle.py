import os
import time
import duckdb
import numpy as np
import pandas as pd

REPO = r"c:\NEW AMAZON\Amazon-ML-Challenge-2026"
NORM_DIR = os.path.join(REPO, "outputs", "person1_step1", "normalized")
P1_OUT = os.path.join(REPO, "outputs", "person1_step1")
REPORT_DIR = os.path.join(REPO, "P2", "reports")
FOLDS_PATH = os.path.join(REPO, "P3", "reports", "folds_v1_manifest.tsv")
GT_PATH = os.path.join(P1_OUT, "train_ground_truth_reconstructed.tsv")
OUT_TSV = os.path.join(REPORT_DIR, "C01_V2_ACCEPTANCE.tsv")

TMP_DIR = os.path.join(REPO, "P2", "data", "duckdb_tmp_v2_oracle")
if os.path.exists(TMP_DIR):
    import shutil
    shutil.rmtree(TMP_DIR)
os.makedirs(TMP_DIR, exist_ok=True)

con = duckdb.connect()
con.execute(f"SET temp_directory='{TMP_DIR}'")
con.execute("SET memory_limit='3GB'")
con.execute("SET threads=2")
con.execute("SET preserve_insertion_order=false")

COMMON_PREFIXES = "('the', 'shri', 'sri', 'dr', 'm/s', 'hotel', 'new', 'om', 'sai', 'jai', 'a', 'an')"

t0 = time.time()
print("1. Loading ground truth and folds...")
con.execute(f"""
    CREATE OR REPLACE TABLE gt_exp AS 
    SELECT source1_entity_id, TRIM(UNNEST(string_split(matched_entity_ids, ','))) AS matched_entity_id
    FROM read_csv('{GT_PATH}', delim='\\t', header=true, all_varchar=true)
    WHERE matched_entity_ids IS NOT NULL AND TRIM(matched_entity_ids) <> ''
""")
con.execute("CREATE OR REPLACE TABLE gt_s2 AS SELECT * FROM gt_exp WHERE matched_entity_id LIKE 'S2-%'")
con.execute("CREATE OR REPLACE TABLE gt_s3 AS SELECT * FROM gt_exp WHERE matched_entity_id LIKE 'S3-%'")

con.execute(f"CREATE TABLE folds AS SELECT source1_entity_id, fold FROM read_csv('{FOLDS_PATH}', delim='\\t', header=true)")
con.execute(f"""
    CREATE TABLE gt_counts AS 
    SELECT 
        source1_entity_id, 
        CASE WHEN matched_entity_ids IS NULL OR matched_entity_ids = '' THEN 0 ELSE len(string_split(matched_entity_ids, ',')) END AS A,
        len(list_filter(string_split(coalesce(matched_entity_ids, ''), ','), x -> x LIKE 'S2-%')) AS A_s2,
        len(list_filter(string_split(coalesce(matched_entity_ids, ''), ','), x -> x LIKE 'S3-%')) AS A_s3
    FROM read_csv('{GT_PATH}', delim='\\t', header=true)
""")

print("2. Loading train_source1...")
con.execute(f"""
    CREATE OR REPLACE TABLE train_s1 AS 
    SELECT entity_id, business_name, business_address, country,
           LEFT(TRIM(business_name), 4) AS prefix4,
           split_part(TRIM(business_name), ' ', 1) AS token1,
           regexp_extract(TRIM(business_address), '[0-9]+[A-Za-z]?', 0) AS house_v1,
           regexp_replace(regexp_extract(TRIM(business_address), '[0-9]+[A-Za-z]?', 0), '^0+', '') AS house_norm,
           regexp_replace(lower(trim(business_name)), '[^a-z0-9]', '', 'g') AS clean_name,
           regexp_replace(lower(trim(business_address)), '[^a-z0-9]', '', 'g') AS clean_addr,
           LEFT(TRIM(business_name), 3) AS prefix3,
           CASE 
               WHEN lower(split_part(TRIM(business_name), ' ', 1)) IN {COMMON_PREFIXES} 
                    AND split_part(TRIM(business_name), ' ', 2) <> '' 
               THEN split_part(TRIM(business_name), ' ', 2)
               ELSE split_part(TRIM(business_name), ' ', 1)
           END AS root_token,
           regexp_extract(TRIM(business_address), '(?:^|[^0-9])([0-9]{{5,6}})(?:[^0-9]|$)', 1) AS zip_code
    FROM read_csv('{NORM_DIR}/train_source1_normalized.tsv', delim='\\t', header=true, all_varchar=true)
""")

# ==============================================================================
# S2 Processing
# ==============================================================================
print("\n3. Loading train_source2...")
t_s2 = time.time()
con.execute(f"""
    CREATE OR REPLACE TABLE train_s2 AS 
    SELECT entity_id, business_name, business_address, country,
           LEFT(TRIM(business_name), 4) AS prefix4,
           split_part(TRIM(business_name), ' ', 1) AS token1,
           regexp_extract(TRIM(business_address), '[0-9]+[A-Za-z]?', 0) AS house_v1,
           regexp_replace(regexp_extract(TRIM(business_address), '[0-9]+[A-Za-z]?', 0), '^0+', '') AS house_norm,
           regexp_replace(lower(trim(business_name)), '[^a-z0-9]', '', 'g') AS clean_name,
           regexp_replace(lower(trim(business_address)), '[^a-z0-9]', '', 'g') AS clean_addr,
           LEFT(TRIM(business_name), 3) AS prefix3,
           CASE 
               WHEN lower(split_part(TRIM(business_name), ' ', 1)) IN {COMMON_PREFIXES} 
                    AND split_part(TRIM(business_name), ' ', 2) <> '' 
               THEN split_part(TRIM(business_name), ' ', 2)
               ELSE split_part(TRIM(business_name), ' ', 1)
           END AS root_token,
           regexp_extract(TRIM(business_address), '(?:^|[^0-9])([0-9]{{5,6}})(?:[^0-9]|$)', 1) AS zip_code
    FROM read_csv('{NORM_DIR}/train_source2_normalized.tsv', delim='\\t', header=true, all_varchar=true)
""")

print("  Generating and aggregating train_s2_v2...")
con.execute("""
    CREATE OR REPLACE TABLE train_s2_v2_raw AS
    -- Rule A
    SELECT s1.entity_id as source1_entity_id, tgt.entity_id as matched_entity_id
    FROM train_s1 s1 JOIN train_s2 tgt 
    ON s1.country <> '' AND s1.country = tgt.country 
       AND s1.prefix4 <> '' AND s1.prefix4 = tgt.prefix4 
       AND s1.house_v1 <> '' AND s1.house_v1 = tgt.house_v1
    UNION
    -- Rule B
    SELECT s1.entity_id, tgt.entity_id
    FROM train_s1 s1 JOIN train_s2 tgt 
    ON s1.country <> '' AND s1.country = tgt.country 
       AND s1.business_address <> '' AND s1.business_address = tgt.business_address
    UNION
    -- Rule C
    SELECT s1.entity_id, tgt.entity_id
    FROM train_s1 s1 JOIN train_s2 tgt 
    ON s1.country <> '' AND s1.country = tgt.country 
       AND s1.token1 <> '' AND s1.token1 = tgt.token1 
       AND s1.house_v1 <> '' AND s1.house_v1 = tgt.house_v1
    UNION
    -- Rule D
    SELECT s1.entity_id, tgt.entity_id
    FROM train_s1 s1 JOIN train_s2 tgt 
    ON s1.country <> '' AND s1.country = tgt.country 
       AND TRIM(s1.business_name) <> '' AND TRIM(s1.business_name) = TRIM(tgt.business_name)
    UNION
    -- Rule E
    SELECT s1.entity_id, tgt.entity_id
    FROM train_s1 s1 JOIN train_s2 tgt 
    ON s1.country <> '' AND s1.country = tgt.country 
       AND s1.clean_name <> '' AND LENGTH(s1.clean_name) >= 3 AND s1.clean_name = tgt.clean_name
    UNION
    -- Rule F1
    SELECT s1.entity_id, tgt.entity_id
    FROM train_s1 s1 JOIN train_s2 tgt 
    ON s1.country <> '' AND s1.country = tgt.country 
       AND s1.prefix4 <> '' AND s1.prefix4 = tgt.prefix4 
       AND s1.house_norm <> '' AND s1.house_norm = tgt.house_norm
    UNION
    -- Rule F2
    SELECT s1.entity_id, tgt.entity_id
    FROM train_s1 s1 JOIN train_s2 tgt 
    ON s1.country <> '' AND s1.country = tgt.country 
       AND s1.token1 <> '' AND s1.token1 = tgt.token1 
       AND s1.house_norm <> '' AND s1.house_norm = tgt.house_norm
    UNION
    -- Rule G
    SELECT s1.entity_id, tgt.entity_id
    FROM train_s1 s1 JOIN train_s2 tgt 
    ON s1.country <> '' AND s1.country = tgt.country 
       AND s1.root_token <> '' AND LENGTH(s1.root_token) >= 3 AND s1.root_token = tgt.root_token 
       AND s1.house_norm <> '' AND s1.house_norm = tgt.house_norm
    UNION
    -- Rule H
    SELECT s1.entity_id, tgt.entity_id
    FROM train_s1 s1 JOIN train_s2 tgt 
    ON s1.country <> '' AND s1.country = tgt.country 
       AND s1.clean_addr <> '' AND LENGTH(s1.clean_addr) >= 6 AND s1.clean_addr = tgt.clean_addr
    UNION
    -- Rule I
    SELECT s1.entity_id, tgt.entity_id
    FROM train_s1 s1 JOIN train_s2 tgt 
    ON s1.country <> '' AND s1.country = tgt.country 
       AND s1.zip_code <> '' AND LENGTH(s1.zip_code) >= 5 AND s1.zip_code = tgt.zip_code 
       AND s1.prefix3 <> '' AND LENGTH(s1.prefix3) = 3 AND s1.prefix3 = tgt.prefix3
""")

s2_cnt = con.execute("SELECT count(*) FROM train_s2_v2_raw").fetchone()[0]
print(f"  train_s2_v2_raw: {s2_cnt:,} rows")

print("  Aggregating S2 counts per S1...")
con.execute("""
    CREATE OR REPLACE TABLE v2_s2_agg AS
    SELECT 
        c.source1_entity_id,
        count(*) AS cands_s2,
        sum(case when gt.matched_entity_id is not null then 1 else 0 end) AS k_s2
    FROM train_s2_v2_raw c
    LEFT JOIN gt_s2 gt ON c.source1_entity_id = gt.source1_entity_id AND c.matched_entity_id = gt.matched_entity_id
    GROUP BY c.source1_entity_id
""")

con.execute("DROP TABLE train_s2_v2_raw")
con.execute("DROP TABLE train_s2")
print(f"  S2 processing done in {time.time() - t_s2:.2f}s")

# ==============================================================================
# S3 Processing
# ==============================================================================
print("\n4. Loading train_source3...")
t_s3 = time.time()
con.execute(f"""
    CREATE OR REPLACE TABLE train_s3 AS 
    SELECT entity_id, business_name, business_address, country,
           LEFT(TRIM(business_name), 4) AS prefix4,
           split_part(TRIM(business_name), ' ', 1) AS token1,
           regexp_extract(TRIM(business_address), '[0-9]+[A-Za-z]?', 0) AS house_v1,
           regexp_replace(regexp_extract(TRIM(business_address), '[0-9]+[A-Za-z]?', 0), '^0+', '') AS house_norm,
           regexp_replace(lower(trim(business_name)), '[^a-z0-9]', '', 'g') AS clean_name,
           regexp_replace(lower(trim(business_address)), '[^a-z0-9]', '', 'g') AS clean_addr,
           LEFT(TRIM(business_name), 3) AS prefix3,
           CASE 
               WHEN lower(split_part(TRIM(business_name), ' ', 1)) IN {COMMON_PREFIXES} 
                    AND split_part(TRIM(business_name), ' ', 2) <> '' 
               THEN split_part(TRIM(business_name), ' ', 2)
               ELSE split_part(TRIM(business_name), ' ', 1)
           END AS root_token,
           regexp_extract(TRIM(business_address), '(?:^|[^0-9])([0-9]{{5,6}})(?:[^0-9]|$)', 1) AS zip_code
    FROM read_csv('{NORM_DIR}/train_source3_normalized.tsv', delim='\\t', header=true, all_varchar=true)
""")

print("  Generating and aggregating train_s3_v2...")
con.execute("""
    CREATE OR REPLACE TABLE train_s3_v2_raw AS
    -- Rule A
    SELECT s1.entity_id as source1_entity_id, tgt.entity_id as matched_entity_id
    FROM train_s1 s1 JOIN train_s3 tgt 
    ON s1.country <> '' AND s1.country = tgt.country 
       AND s1.prefix4 <> '' AND s1.prefix4 = tgt.prefix4 
       AND s1.house_v1 <> '' AND s1.house_v1 = tgt.house_v1
    UNION
    -- Rule B
    SELECT s1.entity_id, tgt.entity_id
    FROM train_s1 s1 JOIN train_s3 tgt 
    ON s1.country <> '' AND s1.country = tgt.country 
       AND s1.business_address <> '' AND s1.business_address = tgt.business_address
    UNION
    -- Rule C
    SELECT s1.entity_id, tgt.entity_id
    FROM train_s1 s1 JOIN train_s3 tgt 
    ON s1.country <> '' AND s1.country = tgt.country 
       AND s1.token1 <> '' AND s1.token1 = tgt.token1 
       AND s1.house_v1 <> '' AND s1.house_v1 = tgt.house_v1
    UNION
    -- Rule D
    SELECT s1.entity_id, tgt.entity_id
    FROM train_s1 s1 JOIN train_s3 tgt 
    ON s1.country <> '' AND s1.country = tgt.country 
       AND TRIM(s1.business_name) <> '' AND TRIM(s1.business_name) = TRIM(tgt.business_name)
    UNION
    -- Rule E
    SELECT s1.entity_id, tgt.entity_id
    FROM train_s1 s1 JOIN train_s3 tgt 
    ON s1.country <> '' AND s1.country = tgt.country 
       AND s1.clean_name <> '' AND LENGTH(s1.clean_name) >= 3 AND s1.clean_name = tgt.clean_name
    UNION
    -- Rule F1
    SELECT s1.entity_id, tgt.entity_id
    FROM train_s1 s1 JOIN train_s3 tgt 
    ON s1.country <> '' AND s1.country = tgt.country 
       AND s1.prefix4 <> '' AND s1.prefix4 = tgt.prefix4 
       AND s1.house_norm <> '' AND s1.house_norm = tgt.house_norm
    UNION
    -- Rule F2
    SELECT s1.entity_id, tgt.entity_id
    FROM train_s1 s1 JOIN train_s3 tgt 
    ON s1.country <> '' AND s1.country = tgt.country 
       AND s1.token1 <> '' AND s1.token1 = tgt.token1 
       AND s1.house_norm <> '' AND s1.house_norm = tgt.house_norm
    UNION
    -- Rule G
    SELECT s1.entity_id, tgt.entity_id
    FROM train_s1 s1 JOIN train_s3 tgt 
    ON s1.country <> '' AND s1.country = tgt.country 
       AND s1.root_token <> '' AND LENGTH(s1.root_token) >= 3 AND s1.root_token = tgt.root_token 
       AND s1.house_norm <> '' AND s1.house_norm = tgt.house_norm
    UNION
    -- Rule H
    SELECT s1.entity_id, tgt.entity_id
    FROM train_s1 s1 JOIN train_s3 tgt 
    ON s1.country <> '' AND s1.country = tgt.country 
       AND s1.clean_addr <> '' AND LENGTH(s1.clean_addr) >= 6 AND s1.clean_addr = tgt.clean_addr
    UNION
    -- Rule I
    SELECT s1.entity_id, tgt.entity_id
    FROM train_s1 s1 JOIN train_s3 tgt 
    ON s1.country <> '' AND s1.country = tgt.country 
       AND s1.zip_code <> '' AND LENGTH(s1.zip_code) >= 5 AND s1.zip_code = tgt.zip_code 
       AND s1.prefix3 <> '' AND LENGTH(s1.prefix3) = 3 AND s1.prefix3 = tgt.prefix3
""")

s3_cnt = con.execute("SELECT count(*) FROM train_s3_v2_raw").fetchone()[0]
print(f"  train_s3_v2_raw: {s3_cnt:,} rows")

print("  Aggregating S3 counts per S1...")
con.execute("""
    CREATE OR REPLACE TABLE v2_s3_agg AS
    SELECT 
        c.source1_entity_id,
        count(*) AS cands_s3,
        sum(case when gt.matched_entity_id is not null then 1 else 0 end) AS k_s3
    FROM train_s3_v2_raw c
    LEFT JOIN gt_s3 gt ON c.source1_entity_id = gt.source1_entity_id AND c.matched_entity_id = gt.matched_entity_id
    GROUP BY c.source1_entity_id
""")

con.execute("DROP TABLE train_s3_v2_raw")
con.execute("DROP TABLE train_s3")
con.execute("DROP TABLE train_s1")
print(f"  S3 processing done in {time.time() - t_s3:.2f}s")

# ==============================================================================
# Master Summary & Candidate Oracle
# ==============================================================================
print("\n5. Assembling Master Summary and computing Candidate Oracle...")
con.execute("""
    CREATE OR REPLACE TABLE s1_v2_summary AS
    SELECT 
        f.source1_entity_id,
        f.fold,
        coalesce(g.A, 0) AS A,
        coalesce(g.A_s2, 0) AS A_s2,
        coalesce(g.A_s3, 0) AS A_s3,
        coalesce(s2.cands_s2, 0) AS cands_s2,
        coalesce(s3.cands_s3, 0) AS cands_s3,
        coalesce(s2.cands_s2, 0) + coalesce(s3.cands_s3, 0) AS total_cands_v2,
        coalesce(s2.k_s2, 0) AS k_s2,
        coalesce(s3.k_s3, 0) AS k_s3,
        coalesce(s2.k_s2, 0) + coalesce(s3.k_s3, 0) AS k_v2,
        -- Candidate Oracle Macro F0.5
        CASE 
            WHEN coalesce(g.A, 0) = 0 THEN 1.0
            WHEN coalesce(s2.k_s2, 0) + coalesce(s3.k_s3, 0) = 0 THEN 0.0
            ELSE (5.0 * (coalesce(s2.k_s2, 0) + coalesce(s3.k_s3, 0))) / (g.A + 4.0 * (coalesce(s2.k_s2, 0) + coalesce(s3.k_s3, 0)))
        END AS cand_oracle_v2,
        -- S2 diagnostic
        CASE 
            WHEN coalesce(g.A_s2, 0) = 0 THEN 1.0
            WHEN coalesce(s2.k_s2, 0) = 0 THEN 0.0
            ELSE (5.0 * coalesce(s2.k_s2, 0)) / (g.A_s2 + 4.0 * coalesce(s2.k_s2, 0))
        END AS cand_oracle_v2_s2,
        -- S3 diagnostic
        CASE 
            WHEN coalesce(g.A_s3, 0) = 0 THEN 1.0
            WHEN coalesce(s3.k_s3, 0) = 0 THEN 0.0
            ELSE (5.0 * coalesce(s3.k_s3, 0)) / (g.A_s3 + 4.0 * coalesce(s3.k_s3, 0))
        END AS cand_oracle_v2_s3
    FROM folds f
    LEFT JOIN gt_counts g ON f.source1_entity_id = g.source1_entity_id
    LEFT JOIN v2_s2_agg s2 ON f.source1_entity_id = s2.source1_entity_id
    LEFT JOIN v2_s3_agg s3 ON f.source1_entity_id = s3.source1_entity_id
""")

# Overall
v2_res = con.execute("""
    SELECT 
        count(*),
        avg(cand_oracle_v2),
        avg(cand_oracle_v2_s2),
        avg(cand_oracle_v2_s3),
        sum(total_cands_v2),
        sum(k_v2),
        sum(cands_s2),
        sum(k_s2),
        sum(cands_s3),
        sum(k_s3)
    FROM s1_v2_summary
""").fetchone()

print("\n" + "=" * 60)
print("=== V2 CANDIDATE ORACLE RESULTS ===")
print("=" * 60)
print(f"Overall Candidate Oracle V2 (Joint): {v2_res[1]:.9f}")
print(f"Overall Candidate Oracle V2 (S2):    {v2_res[2]:.9f}")
print(f"Overall Candidate Oracle V2 (S3):    {v2_res[3]:.9f}")
print(f"Total V2 Candidates: {v2_res[4]:,} (S2: {v2_res[6]:,}, S3: {v2_res[8]:,})")
print(f"Total V2 Captured True Pairs: {v2_res[5]:,} (S2: {v2_res[7]:,}, S3: {v2_res[9]:,})")

# Per fold
v2_folds = con.execute("""
    SELECT 
        fold,
        count(*),
        avg(cand_oracle_v2),
        avg(cand_oracle_v2_s2),
        avg(cand_oracle_v2_s3)
    FROM s1_v2_summary
    GROUP BY fold
    ORDER BY fold
""").fetchall()

print("\nPer-Fold Candidate Oracle V2:")
for f in v2_folds:
    print(f"  Fold {f[0]}: Joint={f[2]:.9f}, S2={f[3]:.9f}, S3={f[4]:.9f} (n={f[1]:,})")

# Capture buckets
buckets = con.execute("""
    SELECT 
        CASE 
            WHEN A = 0 THEN 'TRUE_NO_MATCH'
            WHEN k_v2 = 0 THEN 'ZERO_CAPTURED'
            WHEN k_v2 < A THEN 'PARTIAL_CAPTURED'
            ELSE 'FULL_CAPTURED'
        END AS bucket,
        count(*) AS cnt,
        count(*) * 100.0 / 2206821 AS pct
    FROM s1_v2_summary
    GROUP BY 1
    ORDER BY cnt DESC
""").fetchall()

print("\nV2 S1 Capture Buckets:")
for b in buckets:
    print(f"  {b[0]}: {b[1]:,} ({b[2]:.2f}%)")

# Distribution
dist = con.execute("""
    SELECT 
        min(total_cands_v2),
        percentile_cont(0.25) WITHIN GROUP (ORDER BY total_cands_v2),
        percentile_cont(0.50) WITHIN GROUP (ORDER BY total_cands_v2),
        percentile_cont(0.75) WITHIN GROUP (ORDER BY total_cands_v2),
        percentile_cont(0.90) WITHIN GROUP (ORDER BY total_cands_v2),
        percentile_cont(0.95) WITHIN GROUP (ORDER BY total_cands_v2),
        percentile_cont(0.99) WITHIN GROUP (ORDER BY total_cands_v2),
        max(total_cands_v2),
        avg(total_cands_v2),
        stddev(total_cands_v2)
    FROM s1_v2_summary
""").fetchone()

print("\nV2 Candidates Per S1 Distribution:")
print(f"  Min: {dist[0]}")
print(f"  P25: {dist[1]}")
print(f"  P50 (Median): {dist[2]}")
print(f"  P75: {dist[3]}")
print(f"  P90: {dist[4]}")
print(f"  P95: {dist[5]}")
print(f"  P99: {dist[6]}")
print(f"  Max: {dist[7]}")
print(f"  Mean +/- Std: {dist[8]:.2f} +/- {dist[9]:.2f}")

# Comparison Table
v1_oracle = 0.784522591
v2_oracle = v2_res[1]
oracle_gain = v2_oracle - v1_oracle

print("\n" + "=" * 60)
print(f"V1 Candidate Oracle: {v1_oracle:.9f}")
print(f"V2 Candidate Oracle: {v2_oracle:.9f}")
print(f"ORACLE GAIN:        {oracle_gain:+.9f} ({oracle_gain*100/v1_oracle:+.2f}%)")
print("=" * 60)

# Save TSV
comp_data = [
    {"metric": "Candidate_Oracle_Joint", "v1_value": v1_oracle, "v2_value": v2_oracle, "delta": oracle_gain},
    {"metric": "Candidate_Oracle_S2", "v1_value": 0.714624767, "v2_value": v2_res[2], "delta": v2_res[2] - 0.714624767},
    {"metric": "Candidate_Oracle_S3", "v1_value": 0.724138181, "v2_value": v2_res[3], "delta": v2_res[3] - 0.724138181},
    {"metric": "Total_Candidates", "v1_value": 54592725, "v2_value": v2_res[4], "delta": v2_res[4] - 54592725},
    {"metric": "Captured_Truth_Pairs", "v1_value": 4560579, "v2_value": v2_res[5], "delta": v2_res[5] - 4560579},
    {"metric": "Candidate_Recall_Pct", "v1_value": 59.7062, "v2_value": (v2_res[5] * 100.0 / 7638365), "delta": (v2_res[5] * 100.0 / 7638365) - 59.7062},
    {"metric": "Missed_Truth_Pairs", "v1_value": 3077786, "v2_value": (7638365 - v2_res[5]), "delta": (7638365 - v2_res[5]) - 3077786},
]
df_comp = pd.DataFrame(comp_data)
df_comp.to_csv(OUT_TSV, sep='\t', index=False)
print(f"Saved {OUT_TSV}")

print(f"\nTotal elapsed: {time.time() - t0:.2f}s")
