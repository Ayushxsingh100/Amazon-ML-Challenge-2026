import os
import duckdb

REPO = r"c:\NEW AMAZON\Amazon-ML-Challenge-2026"
FOLDS_PATH = os.path.join(REPO, "P3", "reports", "folds_v1_manifest.tsv")
GT_PATH = os.path.join(REPO, "outputs", "person1_step1", "train_ground_truth_reconstructed.tsv")
OOF_PATH = os.path.join(REPO, "P2", "reports", "E02_OOF_FULL_CANONICAL.tsv")

con = duckdb.connect()
con.execute("SET threads=4")
con.execute("SET memory_limit='4GB'")

print("Checking candidate counts per S1 in full OOF...")
con.execute(f"""
    CREATE TABLE s1_all AS 
    SELECT source1_entity_id, fold 
    FROM read_csv('{FOLDS_PATH}', delim='\\t', header=true)
""")

con.execute(f"""
    CREATE TABLE cand_counts AS
    SELECT 
        source1_entity_id,
        count(*) as total_cands,
        sum(case when matched_entity_id like 'S2-%' then 1 else 0 end) as s2_cands,
        sum(case when matched_entity_id like 'S3-%' then 1 else 0 end) as s3_cands,
        sum(target) as total_pos,
        sum(case when matched_entity_id like 'S2-%' and target=1 then 1 else 0 end) as s2_pos,
        sum(case when matched_entity_id like 'S3-%' and target=1 then 1 else 0 end) as s3_pos
    FROM read_csv('{OOF_PATH}', delim='\\t', header=true)
    GROUP BY source1_entity_id
""")

con.execute(f"""
    CREATE TABLE gt_counts AS
    SELECT 
        source1_entity_id,
        CASE WHEN matched_entity_ids IS NULL OR matched_entity_ids = '' THEN 0 ELSE len(string_split(matched_entity_ids, ',')) END as total_gt,
        len(list_filter(string_split(coalesce(matched_entity_ids, ''), ','), x -> x LIKE 'S2-%')) as s2_gt,
        len(list_filter(string_split(coalesce(matched_entity_ids, ''), ','), x -> x LIKE 'S3-%')) as s3_gt
    FROM read_csv('{GT_PATH}', delim='\\t', header=true)
""")

res = con.execute("""
    SELECT 
        count(*) as total_s1,
        count(c.source1_entity_id) as s1_with_cands,
        count(*) - count(c.source1_entity_id) as s1_zero_cands,
        avg(coalesce(c.total_cands, 0)) as mean_cands,
        percentile_cont(0.50) WITHIN GROUP (ORDER BY coalesce(c.total_cands, 0)) as median_cands,
        max(coalesce(c.total_cands, 0)) as max_cands
    FROM s1_all s
    LEFT JOIN cand_counts c ON s.source1_entity_id = c.source1_entity_id
""").fetchone()

print(f"Total S1: {res[0]}, with cands: {res[1]}, zero cands: {res[2]}")
print(f"Mean cands: {res[3]:.2f}, median: {res[4]}, max: {res[5]}")

gt_res = con.execute("""
    SELECT 
        sum(total_gt) as total_gt_pairs,
        sum(s2_gt) as s2_gt_pairs,
        sum(s3_gt) as s3_gt_pairs,
        count(case when total_gt = 0 then 1 end) as s1_no_gt,
        count(case when total_gt = 1 then 1 end) as s1_gt_1,
        count(case when total_gt >= 2 then 1 end) as s1_gt_2plus,
        count(case when s2_gt > 0 and s3_gt > 0 then 1 end) as s1_both_s2_s3
    FROM gt_counts
""").fetchone()
print(f"GT: total={gt_res[0]}, S2={gt_res[1]}, S3={gt_res[2]}")
print(f"S1 GT size: 0={gt_res[3]}, 1={gt_res[4]}, 2+={gt_res[5]}, both S2+S3={gt_res[6]}")

cap_res = con.execute("""
    SELECT 
        sum(c.total_pos) as total_cap,
        sum(c.s2_pos) as s2_cap,
        sum(c.s3_pos) as s3_cap
    FROM cand_counts c
""").fetchone()
print(f"Captured: total={cap_res[0]}, S2={cap_res[1]}, S3={cap_res[2]}")
