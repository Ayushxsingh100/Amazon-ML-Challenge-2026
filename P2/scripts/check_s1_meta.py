import os
import time
import duckdb
import numpy as np
import pandas as pd

REPO = r"c:\NEW AMAZON\Amazon-ML-Challenge-2026"
FOLDS_PATH = os.path.join(REPO, "P3", "reports", "folds_v1_manifest.tsv")
GT_PATH = os.path.join(REPO, "outputs", "person1_step1", "train_ground_truth_reconstructed.tsv")
S1_PATH = os.path.join(REPO, "outputs", "person1_step1", "normalized", "train_source1_normalized.tsv")
OOF_F0 = os.path.join(REPO, "P2", "reports", "E02_OOF_FULL_CANONICAL_fold0.tsv")

con = duckdb.connect()
con.execute("SET threads=4")
con.execute("SET memory_limit='4GB'")

t0 = time.time()
print("Testing entity stats on Fold 0...")
con.execute(f"""
    CREATE TABLE s1_meta AS
    SELECT 
        entity_id AS source1_entity_id,
        business_name,
        business_address,
        country,
        business_name IS NULL OR TRIM(business_name) = '' OR TRIM(business_name) = 'nan' AS missing_name,
        business_address IS NULL OR TRIM(business_address) = '' OR TRIM(business_address) = 'nan' AS missing_addr,
        regexp_extract(TRIM(business_address), '[0-9]+[A-Za-z]?', 0) AS house,
        (regexp_extract(TRIM(business_address), '[0-9]+[A-Za-z]?', 0) IS NULL OR regexp_extract(TRIM(business_address), '[0-9]+[A-Za-z]?', 0) = '') AS missing_house
    FROM read_csv('{S1_PATH}', delim='\\t', header=true, all_varchar=true)
""")

print(f"Loaded s1_meta: {con.execute('SELECT count(*) FROM s1_meta').fetchone()[0]} rows in {time.time() - t0:.2f}s")
print(con.execute("SELECT missing_name, missing_addr, missing_house, count(*) FROM s1_meta GROUP BY 1, 2, 3").df())
