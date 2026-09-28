import os
import duckdb

repo_root = "c:/NEW AMAZON"
s1_test_path = repo_root + "/Amazon-ML-Challenge-2026/outputs/person1_step1/normalized/test_source1_normalized.tsv"
s2_test_path = repo_root + "/Amazon-ML-Challenge-2026/outputs/person1_step1/normalized/test_source2_normalized.tsv"
s3_test_path = repo_root + "/Amazon-ML-Challenge-2026/outputs/person1_step1/normalized/test_source3_normalized.tsv"
v5_path = repo_root + "/Amazon-ML-Challenge-2026/P2/data/matching_results_v5.tsv"

con = duckdb.connect()
con.execute("PRAGMA threads=8;")
con.execute("PRAGMA memory_limit='24GB';")

print("Checking country counts for target S1s...")
con.execute(f"""
CREATE TEMP TABLE s1 AS SELECT entity_id as s1_id, country FROM read_csv('{s1_test_path}', delim='\t', header=True, all_varchar=True);
CREATE TEMP TABLE s2 AS SELECT entity_id as s2_id, country FROM read_csv('{s2_test_path}', delim='\t', header=True, all_varchar=True);
CREATE TEMP TABLE s3 AS SELECT entity_id as s3_id, country FROM read_csv('{s3_test_path}', delim='\t', header=True, all_varchar=True);
CREATE TEMP TABLE v5 AS SELECT source1_entity_id as s1_id, matched_entity_ids as matched_ids FROM read_csv('{v5_path}', delim='\t', header=True, all_varchar=True);
""")

print("S2 country counts:")
print(con.execute("SELECT country, count(*) FROM s2 GROUP BY country").df())

print("S3 country counts:")
print(con.execute("SELECT country, count(*) FROM s3 GROUP BY country").df())

print("Target S1 country counts:")
print(con.execute("""
WITH targets AS (
    SELECT 
        v.s1_id,
        s.country,
        CASE WHEN matched_ids LIKE '%S2-%' THEN 1 ELSE 0 END as has_s2,
        CASE WHEN matched_ids LIKE '%S3-%' THEN 1 ELSE 0 END as has_s3
    FROM v5 v
    JOIN s1 s ON v.s1_id = s.s1_id
    WHERE matched_ids IS NOT NULL AND TRIM(matched_ids) != ''
)
SELECT 
    country,
    COUNT(CASE WHEN has_s2 = 1 AND has_s3 = 0 THEN 1 END) as search_s3,
    COUNT(CASE WHEN has_s3 = 1 AND has_s2 = 0 THEN 1 END) as search_s2
FROM targets
GROUP BY country;
""").df())
