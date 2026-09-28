import duckdb

con = duckdb.connect()
con.execute("""
CREATE TEMP TABLE best_raw AS
SELECT 
    source1_entity_id as s1_id,
    matched_entity_ids
FROM read_csv('Amazon-ML-Challenge-2026/P2/data/matching_results_v1_conservative.tsv', delim='\t', header=True, all_varchar=True)
WHERE matched_entity_ids IS NOT NULL AND TRIM(matched_entity_ids) != ''
LIMIT 20;
""")

for row in con.execute("SELECT * FROM best_raw").fetchall():
    print(row)
