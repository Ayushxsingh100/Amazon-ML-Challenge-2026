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
print("Reading ground truth...")
# S1 with true match count A
con.execute(f"""
    CREATE TABLE s1_all AS 
    SELECT source1_entity_id, fold 
    FROM read_csv('{FOLDS_PATH}', delim='\\t', header=true)
""")
con.execute(f"""
    CREATE TABLE gt_counts AS 
    SELECT 
        source1_entity_id, 
        CASE 
            WHEN matched_entity_ids IS NULL OR matched_entity_ids = '' THEN 0 
            ELSE len(string_split(matched_entity_ids, ',')) 
        END AS A
    FROM read_csv('{GT_PATH}', delim='\\t', header=true)
""")

print("Aggregating fold 0 predictions at T=0.50...")
con.execute(f"""
    CREATE TABLE f0_preds AS
    SELECT 
        source1_entity_id,
        count(*) AS P,
        sum(target) AS TP
    FROM read_csv('{OOF_F0}', delim='\\t', header=true)
    WHERE oof_score >= 0.50
    GROUP BY source1_entity_id
""")

print("Joining and scoring fold 0...")
query = """
    SELECT 
        f.source1_entity_id,
        coalesce(g.A, 0) AS A,
        coalesce(p.P, 0) AS P,
        coalesce(p.TP, 0) AS TP,
        CASE 
            WHEN coalesce(g.A, 0) = 0 THEN 
                CASE WHEN coalesce(p.P, 0) = 0 THEN 1.0 ELSE 0.0 END
            ELSE 
                CASE WHEN coalesce(p.TP, 0) = 0 THEN 0.0 
                ELSE (5.0 * p.TP) / (g.A + 4.0 * p.P) END
        END AS f05
    FROM s1_all f
    LEFT JOIN gt_counts g ON f.source1_entity_id = g.source1_entity_id
    LEFT JOIN f0_preds p ON f.source1_entity_id = p.source1_entity_id
    WHERE f.fold = 0
"""
df = con.execute(query).fetch_arrow_table().to_pandas()
print(f"Fold 0 evaluated: {len(df)} S1 entities")
print(f"Fold 0 Macro F0.5: {df['f05'].mean():.9f}")
print(f"Elapsed: {time.time() - t0:.2f}s")
