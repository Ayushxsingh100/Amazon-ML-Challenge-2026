import os
import time
import duckdb
import numpy as np

REPO = r"c:\NEW AMAZON\Amazon-ML-Challenge-2026"
FOLDS_PATH = os.path.join(REPO, "P3", "reports", "folds_v1_manifest.tsv")
GT_PATH = os.path.join(REPO, "outputs", "person1_step1", "train_ground_truth_reconstructed.tsv")
OOF_F0 = os.path.join(REPO, "P2", "reports", "E02_OOF_FULL_CANONICAL_fold0.tsv")

con = duckdb.connect()
con.execute("SET threads=4")
con.execute("SET memory_limit='4GB'")

t0 = time.time()
print("Checking fold 0 candidate counts above 0.50...")
con.execute(f"CREATE TABLE s1_f0 AS SELECT source1_entity_id FROM read_csv('{FOLDS_PATH}', delim='\\t', header=true) WHERE fold = 0")
con.execute(f"""
    CREATE TABLE gt_counts AS 
    SELECT 
        source1_entity_id, 
        CASE WHEN matched_entity_ids IS NULL OR matched_entity_ids = '' THEN 0 ELSE len(string_split(matched_entity_ids, ',')) END AS A
    FROM read_csv('{GT_PATH}', delim='\\t', header=true)
""")

con.execute(f"""
    CREATE TABLE cands_f0 AS
    SELECT source1_entity_id, target, oof_score
    FROM read_csv('{OOF_F0}', delim='\\t', header=true)
    WHERE oof_score >= 0.500
""")

res = con.execute("SELECT count(*), count(distinct source1_entity_id) FROM cands_f0").fetchone()
print(f"Candidates >= 0.50 in fold 0: {res[0]} rows, {res[1]} distinct S1 entities")
print(f"Elapsed: {time.time() - t0:.2f}s")
