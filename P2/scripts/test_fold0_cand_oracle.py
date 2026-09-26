import os
import duckdb

REPO = r"c:\NEW AMAZON\Amazon-ML-Challenge-2026"
FOLDS_PATH = os.path.join(REPO, "P3", "reports", "folds_v1_manifest.tsv")
GT_PATH = os.path.join(REPO, "outputs", "person1_step1", "train_ground_truth_reconstructed.tsv")
OOF_F0 = os.path.join(REPO, "P2", "reports", "E02_OOF_FULL_CANONICAL_fold0.tsv")

con = duckdb.connect()
con.execute("SET threads=4")
con.execute(f"CREATE TABLE s1_all AS SELECT source1_entity_id FROM read_csv('{FOLDS_PATH}', delim='\\t', header=true) WHERE fold = 0")
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

con.execute(f"""
    CREATE TABLE cand_pos AS
    SELECT 
        source1_entity_id,
        count(*) AS K
    FROM read_csv('{OOF_F0}', delim='\\t', header=true)
    WHERE target = 1
    GROUP BY source1_entity_id
""")

res = con.execute("""
    SELECT 
        count(*) AS n_entities,
        avg(
            CASE 
                WHEN coalesce(g.A, 0) = 0 THEN 1.0
                ELSE 
                    CASE 
                        WHEN coalesce(c.K, 0) = 0 THEN 0.0
                        ELSE (5.0 * c.K) / (g.A + 4.0 * c.K)
                    END
            END
        ) AS candidate_oracle_macro_f05
    FROM s1_all s
    LEFT JOIN gt_counts g ON s.source1_entity_id = g.source1_entity_id
    LEFT JOIN cand_pos c ON s.source1_entity_id = c.source1_entity_id
""").fetchone()

print(f"Fold 0 Candidate Oracle: {res[1]:.9f}")
