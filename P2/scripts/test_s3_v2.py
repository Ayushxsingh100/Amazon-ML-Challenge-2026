import os
import time
import duckdb
import shutil

REPO = r"c:\NEW AMAZON\Amazon-ML-Challenge-2026"
NORM_DIR = os.path.join(REPO, "outputs", "person1_step1", "normalized")
P1_OUT = os.path.join(REPO, "outputs", "person1_step1")
GT_PATH = os.path.join(P1_OUT, "train_ground_truth_reconstructed.tsv")
TMP_DIR = os.path.join(REPO, "P2", "data", "duckdb_tmp_s3_test")

if os.path.exists(TMP_DIR):
    shutil.rmtree(TMP_DIR)
os.makedirs(TMP_DIR, exist_ok=True)

con = duckdb.connect()
con.execute(f"SET temp_directory='{TMP_DIR}'")
con.execute("SET memory_limit='3GB'")
con.execute("SET threads=2")
con.execute("SET preserve_insertion_order=false")

COMMON_PREFIXES = "('the', 'shri', 'sri', 'dr', 'm/s', 'hotel', 'new', 'om', 'sai', 'jai', 'a', 'an')"

t0 = time.time()
print("1. Loading ground truth S3...")
con.execute(f"""
    CREATE OR REPLACE TABLE gt_exp AS 
    SELECT source1_entity_id, TRIM(UNNEST(string_split(matched_entity_ids, ','))) AS matched_entity_id
    FROM read_csv('{GT_PATH}', delim='\\t', header=true, all_varchar=true)
    WHERE matched_entity_ids IS NOT NULL AND TRIM(matched_entity_ids) <> ''
""")
con.execute("CREATE OR REPLACE TABLE gt_s3 AS SELECT * FROM gt_exp WHERE matched_entity_id LIKE 'S3-%'")
con.execute("DROP TABLE gt_exp")

print("2. Loading train_s1...")
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

print("3. Loading train_s3...")
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

print("4. Generating train_s3_v2_raw...")
t_gen = time.time()
con.execute("""
    CREATE OR REPLACE TABLE train_s3_v2_raw AS
    SELECT s1.entity_id as source1_entity_id, tgt.entity_id as matched_entity_id
    FROM train_s1 s1 JOIN train_s3 tgt 
    ON s1.country <> '' AND s1.country = tgt.country 
       AND s1.prefix4 <> '' AND s1.prefix4 = tgt.prefix4 
       AND s1.house_v1 <> '' AND s1.house_v1 = tgt.house_v1
    UNION
    SELECT s1.entity_id, tgt.entity_id
    FROM train_s1 s1 JOIN train_s3 tgt 
    ON s1.country <> '' AND s1.country = tgt.country 
       AND s1.business_address <> '' AND s1.business_address = tgt.business_address
    UNION
    SELECT s1.entity_id, tgt.entity_id
    FROM train_s1 s1 JOIN train_s3 tgt 
    ON s1.country <> '' AND s1.country = tgt.country 
       AND s1.token1 <> '' AND s1.token1 = tgt.token1 
       AND s1.house_v1 <> '' AND s1.house_v1 = tgt.house_v1
    UNION
    SELECT s1.entity_id, tgt.entity_id
    FROM train_s1 s1 JOIN train_s3 tgt 
    ON s1.country <> '' AND s1.country = tgt.country 
       AND TRIM(s1.business_name) <> '' AND TRIM(s1.business_name) = TRIM(tgt.business_name)
    UNION
    SELECT s1.entity_id, tgt.entity_id
    FROM train_s1 s1 JOIN train_s3 tgt 
    ON s1.country <> '' AND s1.country = tgt.country 
       AND s1.clean_name <> '' AND LENGTH(s1.clean_name) >= 3 AND s1.clean_name = tgt.clean_name
    UNION
    SELECT s1.entity_id, tgt.entity_id
    FROM train_s1 s1 JOIN train_s3 tgt 
    ON s1.country <> '' AND s1.country = tgt.country 
       AND s1.prefix4 <> '' AND s1.prefix4 = tgt.prefix4 
       AND s1.house_norm <> '' AND s1.house_norm = tgt.house_norm
    UNION
    SELECT s1.entity_id, tgt.entity_id
    FROM train_s1 s1 JOIN train_s3 tgt 
    ON s1.country <> '' AND s1.country = tgt.country 
       AND s1.token1 <> '' AND s1.token1 = tgt.token1 
       AND s1.house_norm <> '' AND s1.house_norm = tgt.house_norm
    UNION
    SELECT s1.entity_id, tgt.entity_id
    FROM train_s1 s1 JOIN train_s3 tgt 
    ON s1.country <> '' AND s1.country = tgt.country 
       AND s1.root_token <> '' AND LENGTH(s1.root_token) >= 3 AND s1.root_token = tgt.root_token 
       AND s1.house_norm <> '' AND s1.house_norm = tgt.house_norm
    UNION
    SELECT s1.entity_id, tgt.entity_id
    FROM train_s1 s1 JOIN train_s3 tgt 
    ON s1.country <> '' AND s1.country = tgt.country 
       AND s1.clean_addr <> '' AND LENGTH(s1.clean_addr) >= 6 AND s1.clean_addr = tgt.clean_addr
    UNION
    SELECT s1.entity_id, tgt.entity_id
    FROM train_s1 s1 JOIN train_s3 tgt 
    ON s1.country <> '' AND s1.country = tgt.country 
       AND s1.zip_code <> '' AND LENGTH(s1.zip_code) >= 5 AND s1.zip_code = tgt.zip_code 
       AND s1.prefix3 <> '' AND LENGTH(s1.prefix3) = 3 AND s1.prefix3 = tgt.prefix3
""")
print(f"train_s3_v2_raw generated in {time.time() - t_gen:.2f}s")

res = con.execute("SELECT count(*), count(distinct source1_entity_id) FROM train_s3_v2_raw").fetchone()
print(f"train_s3_v2: rows={res[0]:,}, distinct_s1={res[1]:,}")

pos = con.execute("""
    SELECT count(*) 
    FROM train_s3_v2_raw c
    JOIN gt_s3 gt ON c.source1_entity_id = gt.source1_entity_id AND c.matched_entity_id = gt.matched_entity_id
""").fetchone()[0]
print(f"train_s3_v2 positives: {pos:,}")

con.execute("DROP TABLE train_s3")
con.execute("DROP TABLE train_s1")
con.execute("DROP TABLE gt_s3")

print("Aggregating S3 per S1 into Parquet...")
out_parquet = os.path.join(REPO, "P2", "data", "v2_s3_agg.parquet")
con.execute(f"""
    COPY (
        SELECT 
            c.source1_entity_id,
            count(*) AS cands_s3,
            sum(case when gt.matched_entity_id is not null then 1 else 0 end) AS k_s3
        FROM train_s3_v2_raw c
        LEFT JOIN (
            SELECT source1_entity_id, TRIM(UNNEST(string_split(matched_entity_ids, ','))) AS matched_entity_id
            FROM read_csv('{GT_PATH}', delim='\\t', header=true, all_varchar=true)
            WHERE matched_entity_ids IS NOT NULL AND TRIM(matched_entity_ids) <> '' AND matched_entity_ids LIKE '%S3-%'
        ) gt ON c.source1_entity_id = gt.source1_entity_id AND c.matched_entity_id = gt.matched_entity_id
        GROUP BY c.source1_entity_id
    ) TO '{out_parquet}' (FORMAT PARQUET)
""")

con.close()
if os.path.exists(TMP_DIR):
    shutil.rmtree(TMP_DIR)
print(f"S3 completed in {time.time() - t0:.2f}s")
