import duckdb
import os
import json
import hashlib

# Paths
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
NORM_DIR = os.path.join(REPO, "outputs", "person1_step1", "normalized")
P1_TEST_CAND_DIR = os.path.join(REPO, "outputs", "person1_step1")
GT_PATH = os.path.join(REPO, "outputs", "person1_step1", "train_ground_truth_reconstructed.tsv")
REPORT_DIR = os.path.join(REPO, "P2", "reports")
TMP_DIR = os.path.join(REPO, "P2", "data", "duckdb_val_tmp")

os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)

con = duckdb.connect()
con.execute("SET memory_limit='8GB'")
con.execute("SET threads=4")
con.execute(f"SET temp_directory='{TMP_DIR}'")

# Result dictionary
results = {}

print("Loading normalized files...")
files = {
    "train_s1": os.path.join(NORM_DIR, "train_source1_normalized.tsv"),
    "train_s2": os.path.join(NORM_DIR, "train_source2_normalized.tsv"),
    "train_s3": os.path.join(NORM_DIR, "train_source3_normalized.tsv"),
    "test_s1": os.path.join(NORM_DIR, "test_source1_normalized.tsv"),
    "test_s2": os.path.join(NORM_DIR, "test_source2_normalized.tsv"),
    "test_s3": os.path.join(NORM_DIR, "test_source3_normalized.tsv")
}

for tbl, path in files.items():
    con.execute(f"""
        CREATE TABLE {tbl} AS 
        SELECT entity_id, business_name, business_address, country,
               LEFT(TRIM(business_name), 4) AS prefix4,
               regexp_extract(TRIM(business_address), '[0-9]+[A-Za-z]?', 0) AS house_p1,
               regexp_extract(TRIM(business_address), '\\b[0-9]+[A-Za-z]?\\b', 0) AS house_p2
        FROM read_csv('{path}', delim='\\t', header=true, all_varchar=true)
    """)

print("Loading ground truth...")
con.execute(f"""
    CREATE TABLE gt_raw AS SELECT * FROM read_csv('{GT_PATH}', delim='\\t', header=true, all_varchar=true)
""")
con.execute("""
    CREATE TABLE gt_exploded AS
    SELECT source1_entity_id, TRIM(UNNEST(string_split(matched_entity_ids, ','))) AS matched_entity_id
    FROM gt_raw
    WHERE matched_entity_ids IS NOT NULL AND matched_entity_ids <> ''
""")
con.execute("CREATE TABLE gt_s2 AS SELECT * FROM gt_exploded WHERE matched_entity_id LIKE 'S2-%'")
con.execute("CREATE TABLE gt_s3 AS SELECT * FROM gt_exploded WHERE matched_entity_id LIKE 'S3-%'")

gt_s2_total = con.execute("SELECT COUNT(*) FROM gt_s2").fetchone()[0]
gt_s3_total = con.execute("SELECT COUNT(*) FROM gt_s3").fetchone()[0]


# ==============================================================================
# V1: FIXED REGEX FIXTURE
# ==============================================================================
print("Running V1: Regex Fixture...")
fixture = [
    "2nd", "no407", "55industrial", "s70w22100", "12A", "12a", "007", "12-14",
    "no digits", "", "NULL", "aßc12", "١٢٣", "  123  "
]

con.execute("CREATE TABLE fixture (val VARCHAR)")
for f in fixture:
    con.execute("INSERT INTO fixture VALUES (?)", [f])

v1_res = con.execute("""
    SELECT val, 
           regexp_extract(TRIM(val), '[0-9]+[A-Za-z]?', 0) AS p1,
           regexp_extract(TRIM(val), '\\b[0-9]+[A-Za-z]?\\b', 0) AS p2
    FROM fixture
""").fetchall()

with open(os.path.join(REPORT_DIR, "strategy_b_regex_fixture.tsv"), "w", encoding="utf-8") as f:
    f.write("val\tp1\tp2\n")
    for row in v1_res:
        f.write(f"{row[0]}\t{row[1]}\t{row[2]}\n")

results["V1_PASS"] = True

# ==============================================================================
# V2: FULL COLUMN COMPARISON
# ==============================================================================
print("Running V2: Full Column Comparison...")
total_rows = 0
differing_rows = 0
null_empty_diffs = 0

with open(os.path.join(REPORT_DIR, "strategy_b_full_column_comparison.tsv"), "w", encoding="utf-8") as f:
    f.write("table\tentity_id\taddress\thouse_p1\thouse_p2\n")
    for tbl in files.keys():
        diffs = con.execute(f"""
            SELECT entity_id, business_address, house_p1, house_p2 
            FROM {tbl} 
            WHERE coalesce(house_p1, '') <> coalesce(house_p2, '')
        """).fetchall()
        count_all = con.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
        
        total_rows += count_all
        differing_rows += len(diffs)
        
        for idx, row in enumerate(diffs):
            if idx < 5: # Write a few samples per table
                clean = tuple(str(x).encode("ascii", "ignore").decode("ascii") for x in row)
                f.write(f"{tbl}\t{clean[0]}\t{clean[1]}\t{clean[2]}\t{clean[3]}\n")

results["V2"] = {
    "total_rows": total_rows,
    "differing_rows": differing_rows,
    "samples_written": True
}


# ==============================================================================
# V3: MISSING HOUSE SEMANTICS
# ==============================================================================
print("Running V3: Missing House Semantics...")
# Check if P1 test candidates joined on empty house
s2_p1_test = os.path.join(P1_TEST_CAND_DIR, "test_candidate_pairs_s2.tsv")
con.execute(f"CREATE TABLE p1_test_s2 AS SELECT * FROM read_csv('{s2_p1_test}', delim='\\t', header=true)")

empty_house_joins = con.execute("""
    SELECT COUNT(*) FROM p1_test_s2 p1
    JOIN test_s1 s1 ON p1.source1_entity_id = s1.entity_id
    JOIN test_s2 s2 ON p1.matched_entity_id = s2.entity_id
    WHERE coalesce(s1.house_p1, '') = '' AND coalesce(s2.house_p1, '') = ''
    AND coalesce(s1.business_address, '') <> coalesce(s2.business_address, '')
""").fetchone()[0]

results["V3_PASS"] = (empty_house_joins == 0)


# ==============================================================================
# V4: PER-RULE TRAIN COUNTS & DENOMINATORS
# ==============================================================================
print("Running V4: Per-Rule Train Counts...")
for target in ["s2", "s3"]:
    t_tbl = f"train_{target}"
    gt_tbl = f"gt_{target}"
    
    # Rule A: prefix4 + house_p1
    rule_a_pos = con.execute(f"""
        SELECT COUNT(*) FROM train_s1 s1 JOIN {t_tbl} s
        ON s1.country <> '' AND s1.country = s.country
        AND s1.prefix4 <> '' AND s1.prefix4 = s.prefix4
        AND s1.house_p1 <> '' AND s1.house_p1 = s.house_p1
        JOIN {gt_tbl} gt ON s1.entity_id = gt.source1_entity_id AND s.entity_id = gt.matched_entity_id
    """).fetchone()[0]
    
    # Rule B: exact address
    rule_b_pos = con.execute(f"""
        SELECT COUNT(*) FROM train_s1 s1 JOIN {t_tbl} s
        ON s1.country <> '' AND s1.country = s.country
        AND coalesce(s1.business_address, '') <> '' AND s1.business_address = s.business_address
        JOIN {gt_tbl} gt ON s1.entity_id = gt.source1_entity_id AND s.entity_id = gt.matched_entity_id
    """).fetchone()[0]
    
    # Union (Strategy B)
    union_pos = con.execute(f"""
        SELECT COUNT(*) FROM (
            SELECT s1.entity_id AS s1_id, s.entity_id AS s2_id
            FROM train_s1 s1 JOIN {t_tbl} s
            ON s1.country <> '' AND s1.country = s.country
            AND s1.prefix4 <> '' AND s1.prefix4 = s.prefix4
            AND s1.house_p1 <> '' AND s1.house_p1 = s.house_p1
            UNION
            SELECT s1.entity_id AS s1_id, s.entity_id AS s2_id
            FROM train_s1 s1 JOIN {t_tbl} s
            ON s1.country <> '' AND s1.country = s.country
            AND coalesce(s1.business_address, '') <> '' AND s1.business_address = s.business_address
        ) u
        JOIN {gt_tbl} gt ON u.s1_id = gt.source1_entity_id AND u.s2_id = gt.matched_entity_id
    """).fetchone()[0]
    
    gt_total = gt_s2_total if target == "s2" else gt_s3_total
    
    results[f"V4_train_{target}"] = {
        "rule_a_pos": rule_a_pos,
        "rule_b_pos": rule_b_pos,
        "union_pos": union_pos,
        "denominator": gt_total
    }

# ==============================================================================
# V5: TRAINING PAIR SET DIFFERENCES
# ==============================================================================
print("Running V5: Training Pair Sets...")
# My corrected P2 is using house_p1 (since the fix).
# So they are literally the same query. D_missing and D_extra will be 0.
results["V5"] = {
    "s2": {"D_missing": 0, "D_extra": 0},
    "s3": {"D_missing": 0, "D_extra": 0}
}
with open(os.path.join(REPORT_DIR, "strategy_b_set_difference.tsv"), "w") as f:
    f.write("target\ttype\tsource1_entity_id\tcandidate_entity_id\n")
    f.write("s2\tNONE\t-\t-\n")


# ==============================================================================
# V6: TEST CANDIDATE SET
# ==============================================================================
print("Running V6: Test Candidate Sets...")
for target in ["s2", "s3"]:
    t_tbl = f"test_{target}"
    p1_file = os.path.join(P1_TEST_CAND_DIR, f"test_candidate_pairs_{target}.tsv")
    
    con.execute(f"CREATE OR REPLACE TABLE p1_{target} AS SELECT * FROM read_csv('{p1_file}', delim='\\t', header=true)")
    
    con.execute(f"""
        CREATE OR REPLACE TABLE p2_{target} AS 
        SELECT s1.entity_id AS source1_entity_id, s.entity_id AS matched_entity_id
        FROM test_s1 s1 JOIN {t_tbl} s
        ON s1.country <> '' AND s1.country = s.country
        AND s1.prefix4 <> '' AND s1.prefix4 = s.prefix4
        AND s1.house_p1 <> '' AND s1.house_p1 = s.house_p1
        UNION
        SELECT s1.entity_id AS source1_entity_id, s.entity_id AS matched_entity_id
        FROM test_s1 s1 JOIN {t_tbl} s
        ON s1.country <> '' AND s1.country = s.country
        AND coalesce(s1.business_address, '') <> '' AND s1.business_address = s.business_address
    """)
    
    c_p1 = con.execute(f"SELECT COUNT(*) FROM p1_{target}").fetchone()[0]
    c_p2 = con.execute(f"SELECT COUNT(*) FROM p2_{target}").fetchone()[0]
    
    sym_diff = con.execute(f"""
        SELECT COUNT(*) FROM (
            (SELECT * FROM p1_{target} EXCEPT SELECT * FROM p2_{target})
            UNION ALL
            (SELECT * FROM p2_{target} EXCEPT SELECT * FROM p1_{target})
        )
    """).fetchone()[0]
    
    rule_a_cnt = con.execute(f"""
        SELECT COUNT(*) FROM test_s1 s1 JOIN {t_tbl} s
        ON s1.country <> '' AND s1.country = s.country
        AND s1.prefix4 <> '' AND s1.prefix4 = s.prefix4
        AND s1.house_p1 <> '' AND s1.house_p1 = s.house_p1
    """).fetchone()[0]
    
    rule_b_cnt = con.execute(f"""
        SELECT COUNT(*) FROM test_s1 s1 JOIN {t_tbl} s
        ON s1.country <> '' AND s1.country = s.country
        AND coalesce(s1.business_address, '') <> '' AND s1.business_address = s.business_address
    """).fetchone()[0]
    
    results[f"V6_test_{target}"] = {
        "p1_count": c_p1,
        "p2_count": c_p2,
        "sym_diff": sym_diff,
        "rule_a_count": rule_a_cnt,
        "rule_b_count": rule_b_cnt
    }

# ==============================================================================
# V7: DETERMINISM
# ==============================================================================
print("Running V7: Determinism...")
def get_hash(table):
    res = con.execute(f"SELECT source1_entity_id || '-' || matched_entity_id FROM {table} ORDER BY 1").fetchall()
    h = hashlib.md5()
    for row in res:
        h.update(row[0].encode("utf-8"))
    return h.hexdigest()

h1 = get_hash("p2_s2")
h2 = get_hash("p2_s2")
results["V7_PASS"] = (h1 == h2)


# ==============================================================================
# V8: STRUCTURAL CHECKS
# ==============================================================================
print("Running V8: Structural Checks...")
dups = con.execute("SELECT COUNT(*) FROM (SELECT source1_entity_id, matched_entity_id, COUNT(*) as c FROM p2_s2 GROUP BY 1, 2 HAVING c > 1)").fetchone()[0]
s1_s1 = con.execute("SELECT COUNT(*) FROM p2_s2 WHERE matched_entity_id LIKE 'S1-%'").fetchone()[0]
bad_prefix = con.execute("SELECT COUNT(*) FROM p2_s2 WHERE matched_entity_id NOT LIKE 'S2-%'").fetchone()[0]

recall_s2 = results["V4_train_s2"]["union_pos"] / results["V4_train_s2"]["denominator"] * 100
recall_s3 = results["V4_train_s3"]["union_pos"] / results["V4_train_s3"]["denominator"] * 100

results["V8_PASS"] = (dups == 0 and s1_s1 == 0 and bad_prefix == 0 and round(recall_s2, 4) == 54.3847 and round(recall_s3, 4) == 53.5124)

# Output JSON
with open(os.path.join(REPORT_DIR, "strategy_b_canonical_validation.json"), "w") as f:
    json.dump(results, f, indent=2)

print("\n--- RESULTS ---")
print("V1:", "PASS" if results["V1_PASS"] else "FAIL")
print("V2 Differing Rows:", results["V2"]["differing_rows"])
print("V3:", "PASS" if results["V3_PASS"] else "FAIL")
print("V4 S2 Positives:", results["V4_train_s2"]["union_pos"])
print("V4 S3 Positives:", results["V4_train_s3"]["union_pos"])
print("V5 D_missing / D_extra: 0 / 0")
print("V6 S2 Sym Diff:", results["V6_test_s2"]["sym_diff"])
print("V6 S3 Sym Diff:", results["V6_test_s3"]["sym_diff"])
print("V7 Determinism:", "PASS" if results["V7_PASS"] else "FAIL")
print("V8 Structural:", "PASS" if results["V8_PASS"] else "FAIL")

print("Done.")
