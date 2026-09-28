import duckdb

con = duckdb.connect()
row = con.execute("""
SELECT source1_entity_id, matched_entity_ids 
FROM read_csv('Amazon-ML-Challenge-2026/P2/data/matching_results_v1_conservative.tsv', delim='\t', header=True, all_varchar=True)
WHERE matched_entity_ids IS NULL OR TRIM(matched_entity_ids) = ''
LIMIT 5;
""").fetchall()
print("Sample empty rows from DuckDB:")
print(row)

# Let's inspect the raw bytes of an empty row from the file directly:
with open('Amazon-ML-Challenge-2026/P2/data/matching_results_v1_conservative.tsv', 'rb') as f:
    header = f.readline()
    print("Header bytes:", header)
    count = 0
    for line in f:
        parts = line.split(b'\t')
        if len(parts) == 1 or parts[1].strip() == b'':
            print("Empty row sample bytes:", line)
            count += 1
            if count >= 3:
                break
