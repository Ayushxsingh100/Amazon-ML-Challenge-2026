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
print("Loading ground truth counts and folds...")
con.execute(f"""
    CREATE TABLE s1_all AS 
    SELECT source1_entity_id, fold 
    FROM read_csv('{FOLDS_PATH}', delim='\\t', header=true)
    WHERE fold = 0
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

print("Ranking candidates for Fold 0...")
# In fold 0, rank candidates by oof_score descending
# For each positive (target=1), compute rank pos_j and j (row_number among target=1)
con.execute(f"""
    CREATE TABLE ranked_pos AS
    WITH ranked AS (
        SELECT 
            source1_entity_id,
            target,
            row_number() OVER (PARTITION BY source1_entity_id ORDER BY oof_score DESC) AS rank
        FROM read_csv('{OOF_F0}', delim='\\t', header=true)
    )
    SELECT 
        source1_entity_id,
        rank AS pos_j,
        row_number() OVER (PARTITION BY source1_entity_id ORDER BY rank ASC) AS j
    FROM ranked
    WHERE target = 1
""")

print("Computing max F0.5 per S1...")
# For each S1 with at least 1 captured positive:
# max over j of (5.0 * j) / (A + 4.0 * pos_j)
con.execute("""
    CREATE TABLE s1_ranking_oracle AS
    SELECT 
        r.source1_entity_id,
        max((5.0 * r.j) / (g.A + 4.0 * r.pos_j)) AS max_f05
    FROM ranked_pos r
    JOIN gt_counts g ON r.source1_entity_id = g.source1_entity_id
    GROUP BY r.source1_entity_id
""")

print("Joining across all S1 in Fold 0...")
res = con.execute("""
    SELECT 
        count(*) AS n_entities,
        avg(
            CASE 
                WHEN coalesce(g.A, 0) = 0 THEN 1.0
                ELSE coalesce(ro.max_f05, 0.0)
            END
        ) AS ranking_oracle_macro_f05
    FROM s1_all s
    LEFT JOIN gt_counts g ON s.source1_entity_id = g.source1_entity_id
    LEFT JOIN s1_ranking_oracle ro ON s.source1_entity_id = ro.source1_entity_id
""").fetchone()

print(f"Fold 0 Ranking Oracle: {res[1]:.9f} (entities={res[0]})")
print(f"Elapsed: {time.time() - t0:.2f}s")
