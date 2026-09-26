import os
import time
import duckdb
import numpy as np
import pandas as pd

REPO = r"c:\NEW AMAZON\Amazon-ML-Challenge-2026"
FOLDS_PATH = os.path.join(REPO, "P3", "reports", "folds_v1_manifest.tsv")
GT_PATH = os.path.join(REPO, "outputs", "person1_step1", "train_ground_truth_reconstructed.tsv")
REPORT_DIR = os.path.join(REPO, "P2", "reports")

con = duckdb.connect()
con.execute("SET threads=4")
con.execute("SET memory_limit='4GB'")

t0 = time.time()
print("Reading folds and ground truth...")
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

fold_results = []
all_f05_sums = 0.0
total_entities = 0

for fold in range(5):
    t_f = time.time()
    oof_shard = os.path.join(REPORT_DIR, f"E02_OOF_FULL_CANONICAL_fold{fold}.tsv")
    print(f"Aggregating Fold {fold} from {oof_shard}...")
    
    con.execute(f"""
        CREATE OR REPLACE TABLE fold_preds AS
        SELECT 
            source1_entity_id,
            count(*) AS P,
            sum(target) AS TP
        FROM read_csv('{oof_shard}', delim='\\t', header=true)
        WHERE oof_score >= 0.50
        GROUP BY source1_entity_id
    """)
    
    query = f"""
        SELECT 
            f.source1_entity_id,
            f.fold,
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
        LEFT JOIN fold_preds p ON f.source1_entity_id = p.source1_entity_id
        WHERE f.fold = {fold}
    """
    res = con.execute(f"""
        SELECT 
            count(*) AS n_entities,
            sum(f05) AS sum_f05,
            avg(f05) AS macro_f05,
            stddev(f05) AS std_f05,
            sum(P) AS total_P,
            sum(TP) AS total_TP
        FROM ({query})
    """).fetchone()
    
    n_ent, s_f05, m_f05, std_f05, tot_p, tot_tp = res
    fold_results.append({
        "fold": fold,
        "n_entities": n_ent,
        "macro_f05": m_f05,
        "std_f05": std_f05,
        "sum_f05": s_f05,
        "total_P": tot_p,
        "total_TP": tot_tp
    })
    all_f05_sums += s_f05
    total_entities += n_ent
    print(f"Fold {fold}: n={n_ent}, Macro F0.5={m_f05:.9f}, sum={s_f05:.4f}, elapsed={time.time() - t_f:.2f}s")

overall_macro_f05 = all_f05_sums / total_entities
print("-" * 50)
print(f"TOTAL ENTITIES: {total_entities}")
print(f"OVERALL MACRO F0.5: {overall_macro_f05:.9f}")
print(f"TARGET BASELINE:    0.718765173")
print(f"DIFF:               {abs(overall_macro_f05 - 0.718765173):.12f}")
print(f"Total time: {time.time() - t0:.2f}s")
