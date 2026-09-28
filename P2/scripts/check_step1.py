import duckdb
import time

con = duckdb.connect()
con.execute("PRAGMA threads=8;")
con.execute("PRAGMA memory_limit='20GB';")

print("--- Step 1 Investigation ---")
t0 = time.time()

# Let's check OUR_TEST unique S1s
print("Checking OUR_TEST S1s...")
test_s1_stats = con.execute("""
    SELECT 
        COUNT(*) as total_pairs,
        COUNT(DISTINCT source1_entity_id) as distinct_s1
    FROM (
        SELECT source1_entity_id FROM read_csv('Amazon-ML-Challenge-2026/P2/predictions/phase4/test_predictions_s2.tsv', delim='\t', header=True, columns={'source1_entity_id': 'VARCHAR', 'candidate_entity_id': 'VARCHAR', 'model_score': 'DOUBLE', 'predicted_match_label': 'TINYINT'})
        UNION ALL
        SELECT source1_entity_id FROM read_csv('Amazon-ML-Challenge-2026/P2/predictions/phase4/test_predictions_s3.tsv', delim='\t', header=True, columns={'source1_entity_id': 'VARCHAR', 'candidate_entity_id': 'VARCHAR', 'model_score': 'DOUBLE', 'predicted_match_label': 'TINYINT'})
    )
""").df()
print("OUR_TEST S1 stats:")
print(test_s1_stats)

print(f"Time so far: {time.time() - t0:.2f}s")
