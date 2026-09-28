import duckdb
import time

con = duckdb.connect()
con.execute("PRAGMA threads=8;")
con.execute("PRAGMA memory_limit='20GB';")

print("--- Checking BEST row count and empty rows ---")
t0 = time.time()
best_stats = con.execute("""
    SELECT 
        COUNT(*) as total_rows,
        COUNT(source1_entity_id) as s1_count,
        COUNT(DISTINCT source1_entity_id) as distinct_s1,
        COUNT(CASE WHEN matched_entity_ids IS NULL OR TRIM(matched_entity_ids) = '' THEN 1 END) as empty_rows
    FROM read_csv('Amazon-ML-Challenge-2026/P2/data/matching_results_v1_conservative.tsv', delim='\t', header=True, all_varchar=True)
""").df()
print(best_stats)
print(f"Time: {time.time() - t0:.2f}s")
