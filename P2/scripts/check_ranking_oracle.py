import os
import time
import duckdb

REPO = r"c:\NEW AMAZON\Amazon-ML-Challenge-2026"
FOLDS_PATH = os.path.join(REPO, "P3", "reports", "folds_v1_manifest.tsv")
GT_PATH = os.path.join(REPO, "outputs", "person1_step1", "train_ground_truth_reconstructed.tsv")
REPORT_DIR = os.path.join(REPO, "P2", "reports")

con = duckdb.connect()
con.execute("SET threads=4")
con.execute("SET memory_limit='4GB'")

t0 = time.time()
print("Loading ground truth counts and folds...")
con.execute(f"CREATE TABLE s1_all AS SELECT source1_entity_id, fold FROM read_csv('{FOLDS_PATH}', delim='\\t', header=true)")
con.execute(f"""
    CREATE TABLE gt_counts AS 
    SELECT 
        source1_entity_id, 
        CASE WHEN matched_entity_ids IS NULL OR matched_entity_ids = '' THEN 0 ELSE len(string_split(matched_entity_ids, ',')) END AS A,
        len(list_filter(string_split(coalesce(matched_entity_ids, ''), ','), x -> x LIKE 'S2-%')) AS A_s2,
        len(list_filter(string_split(coalesce(matched_entity_ids, ''), ','), x -> x LIKE 'S3-%')) AS A_s3
    FROM read_csv('{GT_PATH}', delim='\\t', header=true)
""")

fold_results = []
all_sums_joint = 0.0
all_sums_s2 = 0.0
all_sums_s3 = 0.0
total_entities = 0

for fold in range(5):
    t_f = time.time()
    oof_shard = os.path.join(REPORT_DIR, f"E02_OOF_FULL_CANONICAL_fold{fold}.tsv")
    print(f"Ranking Fold {fold}...")
    
    # 1. Joint Ranking
    con.execute(f"""
        CREATE OR REPLACE TABLE ranked_pos_joint AS
        WITH ranked AS (
            SELECT 
                source1_entity_id,
                target,
                row_number() OVER (PARTITION BY source1_entity_id ORDER BY oof_score DESC) AS rank
            FROM read_csv('{oof_shard}', delim='\\t', header=true)
        )
        SELECT 
            source1_entity_id,
            rank AS pos_j,
            row_number() OVER (PARTITION BY source1_entity_id ORDER BY rank ASC) AS j
        FROM ranked
        WHERE target = 1
    """)
    
    # 2. S2 Ranking
    con.execute(f"""
        CREATE OR REPLACE TABLE ranked_pos_s2 AS
        WITH ranked AS (
            SELECT 
                source1_entity_id,
                target,
                row_number() OVER (PARTITION BY source1_entity_id ORDER BY oof_score DESC) AS rank
            FROM read_csv('{oof_shard}', delim='\\t', header=true)
            WHERE matched_entity_id LIKE 'S2-%'
        )
        SELECT 
            source1_entity_id,
            rank AS pos_j,
            row_number() OVER (PARTITION BY source1_entity_id ORDER BY rank ASC) AS j
        FROM ranked
        WHERE target = 1
    """)
    
    # 3. S3 Ranking
    con.execute(f"""
        CREATE OR REPLACE TABLE ranked_pos_s3 AS
        WITH ranked AS (
            SELECT 
                source1_entity_id,
                target,
                row_number() OVER (PARTITION BY source1_entity_id ORDER BY oof_score DESC) AS rank
            FROM read_csv('{oof_shard}', delim='\\t', header=true)
            WHERE matched_entity_id LIKE 'S3-%'
        )
        SELECT 
            source1_entity_id,
            rank AS pos_j,
            row_number() OVER (PARTITION BY source1_entity_id ORDER BY rank ASC) AS j
        FROM ranked
        WHERE target = 1
    """)
    
    con.execute("""
        CREATE OR REPLACE TABLE s1_ro AS
        SELECT 
            r.source1_entity_id,
            max((5.0 * r.j) / (g.A + 4.0 * r.pos_j)) AS max_f05
        FROM ranked_pos_joint r
        JOIN gt_counts g ON r.source1_entity_id = g.source1_entity_id
        GROUP BY r.source1_entity_id
    """)
    
    con.execute("""
        CREATE OR REPLACE TABLE s1_ro_s2 AS
        SELECT 
            r.source1_entity_id,
            max((5.0 * r.j) / (g.A_s2 + 4.0 * r.pos_j)) AS max_f05_s2
        FROM ranked_pos_s2 r
        JOIN gt_counts g ON r.source1_entity_id = g.source1_entity_id
        GROUP BY r.source1_entity_id
    """)
    
    con.execute("""
        CREATE OR REPLACE TABLE s1_ro_s3 AS
        SELECT 
            r.source1_entity_id,
            max((5.0 * r.j) / (g.A_s3 + 4.0 * r.pos_j)) AS max_f05_s3
        FROM ranked_pos_s3 r
        JOIN gt_counts g ON r.source1_entity_id = g.source1_entity_id
        GROUP BY r.source1_entity_id
    """)
    
    res = con.execute(f"""
        SELECT 
            count(*),
            sum(CASE WHEN coalesce(g.A, 0) = 0 THEN 1.0 ELSE coalesce(ro.max_f05, 0.0) END),
            avg(CASE WHEN coalesce(g.A, 0) = 0 THEN 1.0 ELSE coalesce(ro.max_f05, 0.0) END),
            sum(CASE WHEN coalesce(g.A_s2, 0) = 0 THEN 1.0 ELSE coalesce(ro2.max_f05_s2, 0.0) END),
            avg(CASE WHEN coalesce(g.A_s2, 0) = 0 THEN 1.0 ELSE coalesce(ro2.max_f05_s2, 0.0) END),
            sum(CASE WHEN coalesce(g.A_s3, 0) = 0 THEN 1.0 ELSE coalesce(ro3.max_f05_s3, 0.0) END),
            avg(CASE WHEN coalesce(g.A_s3, 0) = 0 THEN 1.0 ELSE coalesce(ro3.max_f05_s3, 0.0) END)
        FROM s1_all s
        LEFT JOIN gt_counts g ON s.source1_entity_id = g.source1_entity_id
        LEFT JOIN s1_ro ro ON s.source1_entity_id = ro.source1_entity_id
        LEFT JOIN s1_ro_s2 ro2 ON s.source1_entity_id = ro2.source1_entity_id
        LEFT JOIN s1_ro_s3 ro3 ON s.source1_entity_id = ro3.source1_entity_id
        WHERE s.fold = {fold}
    """).fetchone()
    
    n, sum_j, avg_j, sum_s2, avg_s2, sum_s3, avg_s3 = res
    fold_results.append({
        "fold": fold, "n": n,
        "avg_joint": avg_j, "avg_s2": avg_s2, "avg_s3": avg_s3
    })
    all_sums_joint += sum_j
    all_sums_s2 += sum_s2
    all_sums_s3 += sum_s3
    total_entities += n
    print(f"Fold {fold}: Joint={avg_j:.9f}, S2={avg_s2:.9f}, S3={avg_s3:.9f}, time={time.time() - t_f:.2f}s")

overall_joint = all_sums_joint / total_entities
overall_s2 = all_sums_s2 / total_entities
overall_s3 = all_sums_s3 / total_entities

print("-" * 50)
print(f"Overall Ranking Oracle Joint: {overall_joint:.9f}")
print(f"Overall Ranking Oracle S2:    {overall_s2:.9f}")
print(f"Overall Ranking Oracle S3:    {overall_s3:.9f}")
print(f"Total time: {time.time() - t0:.2f}s")
