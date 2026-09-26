import os
import duckdb

REPO = r"c:\NEW AMAZON\Amazon-ML-Challenge-2026"
FOLDS_PATH = os.path.join(REPO, "P3", "reports", "folds_v1_manifest.tsv")
GT_PATH = os.path.join(REPO, "outputs", "person1_step1", "train_ground_truth_reconstructed.tsv")
OOF_PATH = os.path.join(REPO, "P2", "reports", "E02_OOF_FULL_CANONICAL.tsv")

con = duckdb.connect()
con.execute("SET threads=4")
con.execute("SET memory_limit='4GB'")

con.execute(f"CREATE TABLE s1_all AS SELECT source1_entity_id, fold FROM read_csv('{FOLDS_PATH}', delim='\\t', header=true)")
con.execute(f"""
    CREATE TABLE gt_counts AS 
    SELECT 
        source1_entity_id,
        CASE WHEN matched_entity_ids IS NULL OR matched_entity_ids = '' THEN 0 ELSE len(string_split(matched_entity_ids, ',')) END as A,
        len(list_filter(string_split(coalesce(matched_entity_ids, ''), ','), x -> x LIKE 'S2-%')) as A_s2,
        len(list_filter(string_split(coalesce(matched_entity_ids, ''), ','), x -> x LIKE 'S3-%')) as A_s3
    FROM read_csv('{GT_PATH}', delim='\\t', header=true)
""")

con.execute(f"""
    CREATE TABLE cand_pos AS
    SELECT 
        source1_entity_id,
        count(*) as total_cands,
        sum(target) as K,
        sum(case when matched_entity_id like 'S2-%' and target=1 then 1 else 0 end) as K_s2,
        sum(case when matched_entity_id like 'S3-%' and target=1 then 1 else 0 end) as K_s3
    FROM read_csv('{OOF_PATH}', delim='\\t', header=true)
    GROUP BY source1_entity_id
""")

con.execute("""
    CREATE TABLE s1_status AS
    SELECT 
        s.source1_entity_id,
        s.fold,
        coalesce(g.A, 0) as A,
        coalesce(g.A_s2, 0) as A_s2,
        coalesce(g.A_s3, 0) as A_s3,
        coalesce(c.total_cands, 0) as total_cands,
        coalesce(c.K, 0) as K,
        coalesce(c.K_s2, 0) as K_s2,
        coalesce(c.K_s3, 0) as K_s3,
        CASE 
            WHEN coalesce(g.A, 0) = 0 THEN 'NO_TRUTH'
            WHEN coalesce(c.K, 0) = 0 THEN 'ZERO_CAPTURED'
            WHEN coalesce(c.K, 0) < g.A THEN 'PARTIAL_CAPTURED'
            ELSE 'FULL_CAPTURED'
        END as capture_status,
        CASE 
            WHEN coalesce(g.A_s2, 0) = 0 THEN 'NO_TRUTH'
            WHEN coalesce(c.K_s2, 0) = 0 THEN 'ZERO_CAPTURED'
            WHEN coalesce(c.K_s2, 0) < g.A_s2 THEN 'PARTIAL_CAPTURED'
            ELSE 'FULL_CAPTURED'
        END as capture_status_s2,
        CASE 
            WHEN coalesce(g.A_s3, 0) = 0 THEN 'NO_TRUTH'
            WHEN coalesce(c.K_s3, 0) = 0 THEN 'ZERO_CAPTURED'
            WHEN coalesce(c.K_s3, 0) < g.A_s3 THEN 'PARTIAL_CAPTURED'
            ELSE 'FULL_CAPTURED'
        END as capture_status_s3
    FROM s1_all s
    LEFT JOIN gt_counts g ON s.source1_entity_id = g.source1_entity_id
    LEFT JOIN cand_pos c ON s.source1_entity_id = c.source1_entity_id
""")

res = con.execute("""
    SELECT capture_status, count(*), count(*) * 100.0 / 2206821
    FROM s1_status
    GROUP BY 1
    ORDER BY 2 DESC
""").fetchall()

print("Joint Capture Status:")
for r in res:
    print(f"  {r[0]}: count={r[1]}, pct={r[2]:.2f}%")

res_s2 = con.execute("""
    SELECT capture_status_s2, count(*), count(*) * 100.0 / 2206821
    FROM s1_status
    GROUP BY 1
    ORDER BY 2 DESC
""").fetchall()

print("S2 Capture Status:")
for r in res_s2:
    print(f"  {r[0]}: count={r[1]}, pct={r[2]:.2f}%")

res_s3 = con.execute("""
    SELECT capture_status_s3, count(*), count(*) * 100.0 / 2206821
    FROM s1_status
    GROUP BY 1
    ORDER BY 2 DESC
""").fetchall()

print("S3 Capture Status:")
for r in res_s3:
    print(f"  {r[0]}: count={r[1]}, pct={r[2]:.2f}%")
