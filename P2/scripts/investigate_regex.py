import duckdb
import os
import time

con = duckdb.connect()
con.execute("SET memory_limit='8GB'")

S1 = "outputs/person1_step1/normalized/train_source1_normalized.tsv"
S2 = "outputs/person1_step1/normalized/train_source2_normalized.tsv"
GT = "outputs/person1_step1/train_ground_truth_reconstructed.tsv"

print("Loading data...")
con.execute(f"""
    CREATE TABLE s1 AS
    SELECT entity_id, business_address, country, LEFT(TRIM(business_name), 4) AS prefix4
    FROM read_csv('{S1}', delim='\\t', header=true, all_varchar=true)
""")
con.execute(f"""
    CREATE TABLE s2 AS
    SELECT entity_id, business_address, country, LEFT(TRIM(business_name), 4) AS prefix4
    FROM read_csv('{S2}', delim='\\t', header=true, all_varchar=true)
""")
con.execute(f"""
    CREATE TABLE gt_raw AS
    SELECT source1_entity_id, matched_entity_ids
    FROM read_csv('{GT}', delim='\\t', header=true, all_varchar=true)
""")
con.execute("""
    CREATE TABLE gt_exploded AS
    SELECT source1_entity_id, TRIM(UNNEST(string_split(matched_entity_ids, ','))) AS matched_entity_id
    FROM gt_raw
    WHERE matched_entity_ids IS NOT NULL AND matched_entity_ids <> ''
""")
con.execute("""
    CREATE TABLE gt_s2 AS
    SELECT DISTINCT source1_entity_id, matched_entity_id
    FROM gt_exploded
    WHERE matched_entity_id LIKE 'S2-%'
""")

print("Testing P1's regex...")
con.execute("""
    CREATE TABLE s1_p1 AS SELECT *, regexp_extract(TRIM(business_address), '[0-9]+[A-Za-z]?', 0) AS house FROM s1
""")
con.execute("""
    CREATE TABLE s2_p1 AS SELECT *, regexp_extract(TRIM(business_address), '[0-9]+[A-Za-z]?', 0) AS house FROM s2
""")
count_p1 = con.execute("""
    SELECT COUNT(*) FROM s1_p1 JOIN s2_p1
    ON s1_p1.country <> '' AND s1_p1.country = s2_p1.country
    AND s1_p1.prefix4 <> '' AND s1_p1.prefix4 = s2_p1.prefix4
    AND s1_p1.house <> '' AND s1_p1.house = s2_p1.house
    JOIN gt_s2 ON s1_p1.entity_id = gt_s2.source1_entity_id AND s2_p1.entity_id = gt_s2.matched_entity_id
""").fetchone()[0]

print("Testing P2's regex...")
con.execute("""
    CREATE TABLE s1_p2 AS SELECT *, regexp_extract(TRIM(business_address), '\\b[0-9]+[A-Za-z]?\\b', 0) AS house FROM s1
""")
con.execute("""
    CREATE TABLE s2_p2 AS SELECT *, regexp_extract(TRIM(business_address), '\\b[0-9]+[A-Za-z]?\\b', 0) AS house FROM s2
""")
count_p2 = con.execute("""
    SELECT COUNT(*) FROM s1_p2 JOIN s2_p2
    ON s1_p2.country <> '' AND s1_p2.country = s2_p2.country
    AND s1_p2.prefix4 <> '' AND s1_p2.prefix4 = s2_p2.prefix4
    AND s1_p2.house <> '' AND s1_p2.house = s2_p2.house
    JOIN gt_s2 ON s1_p2.entity_id = gt_s2.source1_entity_id AND s2_p2.entity_id = gt_s2.matched_entity_id
""").fetchone()[0]

print(f"P1 regex true pairs (Rule A only): {count_p1:,}")
print(f"P2 regex true pairs (Rule A only): {count_p2:,}")

# Let's get 5 discrepancy examples
print("\nDiscrepancy examples where P1 matched but P2 didn't:")
res = con.execute("""
    SELECT
        s1.entity_id, s2.entity_id,
        s1.business_address, s2.business_address,
        regexp_extract(TRIM(s1.business_address), '[0-9]+[A-Za-z]?', 0) AS p1_h1,
        regexp_extract(TRIM(s2.business_address), '[0-9]+[A-Za-z]?', 0) AS p1_h2,
        regexp_extract(TRIM(s1.business_address), '\\b[0-9]+[A-Za-z]?\\b', 0) AS p2_h1,
        regexp_extract(TRIM(s2.business_address), '\\b[0-9]+[A-Za-z]?\\b', 0) AS p2_h2
    FROM s1 JOIN s2
    ON s1.country <> '' AND s1.country = s2.country
    AND s1.prefix4 <> '' AND s1.prefix4 = s2.prefix4
    JOIN gt_s2 ON s1.entity_id = gt_s2.source1_entity_id AND s2.entity_id = gt_s2.matched_entity_id
    WHERE
        (regexp_extract(TRIM(s1.business_address), '[0-9]+[A-Za-z]?', 0) <> ''
         AND regexp_extract(TRIM(s1.business_address), '[0-9]+[A-Za-z]?', 0) = regexp_extract(TRIM(s2.business_address), '[0-9]+[A-Za-z]?', 0))
        AND NOT (
         regexp_extract(TRIM(s1.business_address), '\\b[0-9]+[A-Za-z]?\\b', 0) <> ''
         AND regexp_extract(TRIM(s1.business_address), '\\b[0-9]+[A-Za-z]?\\b', 0) = regexp_extract(TRIM(s2.business_address), '\\b[0-9]+[A-Za-z]?\\b', 0)
        )
    LIMIT 20
""").fetchall()

for row in res:
    clean_row = tuple(str(x).encode("ascii", "ignore").decode("ascii") for x in row)
    print(clean_row)

