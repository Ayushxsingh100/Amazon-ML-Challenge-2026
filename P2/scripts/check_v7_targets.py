import os
import duckdb

repo_root = "c:/NEW AMAZON"
p2_data = os.path.join(repo_root, "Amazon-ML-Challenge-2026", "P2", "data")
v5_path = os.path.join(p2_data, "matching_results_v5.tsv")

con = duckdb.connect()
con.execute("PRAGMA threads=8;")
con.execute("PRAGMA memory_limit='24GB';")

print("--- Step 1: Target Rows in V5 ---")
con.execute(f"""
CREATE TEMP TABLE v5_raw AS
SELECT 
    source1_entity_id as s1_id,
    COALESCE(matched_entity_ids, '') as matched_ids,
    CASE WHEN matched_entity_ids IS NULL OR TRIM(matched_entity_ids) = '' THEN 1 ELSE 0 END as is_empty
FROM read_csv('{v5_path.replace(os.sep, "/")}', delim='\t', header=True, all_varchar=True);
""")

con.execute("""
CREATE TEMP TABLE v5_source_counts AS
SELECT 
    s1_id,
    matched_ids,
    CASE WHEN matched_ids LIKE '%S2-%' THEN 1 ELSE 0 END as has_s2,
    CASE WHEN matched_ids LIKE '%S3-%' THEN 1 ELSE 0 END as has_s3
FROM v5_raw
WHERE is_empty = 0;
""")

stats = con.execute("""
SELECT 
    COUNT(*) as total_nonempty,
    COUNT(CASE WHEN has_s2 = 1 AND has_s3 = 1 THEN 1 END) as both_s2_s3,
    COUNT(CASE WHEN has_s2 = 1 AND has_s3 = 0 THEN 1 END) as s2_only_search_s3,
    COUNT(CASE WHEN has_s3 = 1 AND has_s2 = 0 THEN 1 END) as s3_only_search_s2
FROM v5_source_counts;
""").df()

print(stats)
