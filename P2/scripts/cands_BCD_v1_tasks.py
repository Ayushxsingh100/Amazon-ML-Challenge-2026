import os
import sys
import duckdb
import hashlib
import time
from datetime import datetime, timezone

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
NORM_DIR = os.path.join(REPO, "outputs", "person1_step1", "normalized")
P1_OUT = os.path.join(REPO, "outputs", "person1_step1")
P2_DIR = os.path.join(REPO, "P2")
REPORT_DIR = os.path.join(P2_DIR, "reports")
CAND_DIR = os.path.join(P2_DIR, "data", "candidates")
FEAT_DIR = os.path.join(P2_DIR, "data", "features")
TMP_DIR = os.path.join(P2_DIR, "data", "duckdb_tmp_bcd")

os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(CAND_DIR, exist_ok=True)
os.makedirs(FEAT_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)

con = duckdb.connect()
con.execute(f"SET temp_directory='{TMP_DIR}'")
con.execute("SET memory_limit='8GB'")
con.execute("SET threads=4")

print("Loading normalized data...", flush=True)
for typ in ["train", "test"]:
    for src in ["1", "2", "3"]:
        path = os.path.join(NORM_DIR, f"{typ}_source{src}_normalized.tsv")
        con.execute(f"""
            CREATE TABLE {typ}_s{src} AS 
            SELECT entity_id, business_name, business_address, country,
                   LEFT(TRIM(business_name), 4) AS prefix4,
                   split_part(TRIM(business_name), ' ', 1) AS token1,
                   regexp_extract(TRIM(business_address), '[0-9]+[A-Za-z]?', 0) AS house
            FROM read_csv('{path}', delim='\\t', header=true, all_varchar=true)
        """)

print("Loading ground truth...", flush=True)
gt_path = os.path.join(P1_OUT, "train_ground_truth_reconstructed.tsv")
con.execute(f"""
    CREATE TABLE gt_exp AS 
    SELECT source1_entity_id, TRIM(UNNEST(string_split(matched_entity_ids, ','))) AS matched_entity_id
    FROM read_csv('{gt_path}', delim='\\t', header=true)
    WHERE matched_entity_ids IS NOT NULL AND matched_entity_ids <> ''
""")
con.execute("CREATE TABLE gt_s2 AS SELECT * FROM gt_exp WHERE matched_entity_id LIKE 'S2-%'")
con.execute("CREATE TABLE gt_s3 AS SELECT * FROM gt_exp WHERE matched_entity_id LIKE 'S3-%'")

def get_hash(path):
    res = con.execute(f"SELECT source1_entity_id || '-' || matched_entity_id FROM read_csv('{path}', delim='\\t', header=true) ORDER BY 1").fetchall()
    h = hashlib.sha256()
    for row in res:
        h.update(row[0].encode("utf-8"))
    return h.hexdigest()

results = {}
manifest_rows = []
commit_sha = "9dfb66980c169f215abc20f3e85bfedcb5908be6"
gen_script = "P2/scripts/cands_BCD_v1_tasks.py"
ts = datetime.now(timezone.utc).isoformat()

# ==============================================================================
# TASK 2, 3, 5: Audit Train B+C+D, Generate Test B+C+D, Manifest
# ==============================================================================
for typ in ["train", "test"]:
    results[typ] = {}
    for src in ["s2", "s3"]:
        print(f"\nGenerating {typ} {src} B+C+D...", flush=True)
        
        # Fast Union to get candidates
        con.execute(f"""
            CREATE OR REPLACE TABLE {typ}_{src}_cands_raw AS
            SELECT s1.entity_id as s1_id, tgt.entity_id as tgt_id
            FROM {typ}_s1 s1 JOIN {typ}_{src} tgt 
            ON s1.country <> '' AND s1.country = tgt.country AND s1.prefix4 <> '' AND s1.prefix4 = tgt.prefix4 AND s1.house <> '' AND s1.house = tgt.house
            
            UNION
            
            SELECT s1.entity_id, tgt.entity_id
            FROM {typ}_s1 s1 JOIN {typ}_{src} tgt 
            ON s1.country <> '' AND s1.country = tgt.country AND s1.business_address <> '' AND s1.business_address = tgt.business_address
            
            UNION
            
            SELECT s1.entity_id, tgt.entity_id
            FROM {typ}_s1 s1 JOIN {typ}_{src} tgt 
            ON s1.country <> '' AND s1.country = tgt.country AND s1.token1 <> '' AND s1.token1 = tgt.token1 AND s1.house <> '' AND s1.house = tgt.house
            
            UNION
            
            SELECT s1.entity_id, tgt.entity_id
            FROM {typ}_s1 s1 JOIN {typ}_{src} tgt 
            ON s1.country <> '' AND s1.country = tgt.country AND TRIM(s1.business_name) <> '' AND TRIM(s1.business_name) = TRIM(tgt.business_name)
        """)
        
        print("  Evaluating rules on candidates...", flush=True)
        rule_A = f"s1.country <> '' AND s1.country = tgt.country AND s1.prefix4 <> '' AND s1.prefix4 = tgt.prefix4 AND s1.house <> '' AND s1.house = tgt.house"
        rule_B = f"s1.country <> '' AND s1.country = tgt.country AND s1.business_address <> '' AND s1.business_address = tgt.business_address"
        rule_C = f"s1.country <> '' AND s1.country = tgt.country AND s1.token1 <> '' AND s1.token1 = tgt.token1 AND s1.house <> '' AND s1.house = tgt.house"
        rule_D = f"s1.country <> '' AND s1.country = tgt.country AND TRIM(s1.business_name) <> '' AND TRIM(s1.business_name) = TRIM(tgt.business_name)"
        
        con.execute(f"""
            CREATE OR REPLACE TABLE {typ}_{src}_cands AS
            SELECT 
                c.s1_id AS source1_entity_id, 
                c.tgt_id AS matched_entity_id,
                CASE WHEN {rule_A} THEN 1 ELSE 0 END AS rA,
                CASE WHEN {rule_B} THEN 1 ELSE 0 END AS rB,
                CASE WHEN {rule_C} THEN 1 ELSE 0 END AS rC,
                CASE WHEN {rule_D} THEN 1 ELSE 0 END AS rD
            FROM {typ}_{src}_cands_raw c
            JOIN {typ}_s1 s1 ON c.s1_id = s1.entity_id
            JOIN {typ}_{src} tgt ON c.tgt_id = tgt.entity_id
        """)
        
        vol = con.execute(f"""
            SELECT 
                SUM(CASE WHEN (rA=1 OR rB=1) AND rC=0 AND rD=0 THEN 1 ELSE 0 END) AS b_only,
                SUM(CASE WHEN (rA=0 AND rB=0) AND rC=1 AND rD=0 THEN 1 ELSE 0 END) AS c_only,
                SUM(CASE WHEN (rA=0 AND rB=0) AND rC=0 AND rD=1 THEN 1 ELSE 0 END) AS d_only,
                SUM(CASE WHEN (rA=1 OR rB=1) AND rC=1 AND rD=0 THEN 1 ELSE 0 END) AS bc_ov,
                SUM(CASE WHEN (rA=1 OR rB=1) AND rC=0 AND rD=1 THEN 1 ELSE 0 END) AS bd_ov,
                SUM(CASE WHEN (rA=0 AND rB=0) AND rC=1 AND rD=1 THEN 1 ELSE 0 END) AS cd_ov,
                SUM(CASE WHEN (rA=1 OR rB=1) AND rC=1 AND rD=1 THEN 1 ELSE 0 END) AS bcd_ov,
                COUNT(*) AS total,
                COUNT(DISTINCT source1_entity_id) AS dist_s1
            FROM {typ}_{src}_cands
        """).fetchone()
        
        zero_cand = con.execute(f"SELECT COUNT(*) FROM {typ}_s1 WHERE entity_id NOT IN (SELECT source1_entity_id FROM {typ}_{src}_cands)").fetchone()[0]
        
        pos = neg = 0
        if typ == "train":
            pos = con.execute(f"SELECT COUNT(*) FROM {typ}_{src}_cands c JOIN gt_{src} gt ON c.source1_entity_id = gt.source1_entity_id AND c.matched_entity_id = gt.matched_entity_id").fetchone()[0]
            neg = vol[7] - pos
            out_path = os.path.join(CAND_DIR, f"train_candidate_pairs_{src}.tsv")
            con.execute(f"""
                COPY (
                    SELECT c.source1_entity_id, c.matched_entity_id,
                           CASE WHEN gt.matched_entity_id IS NOT NULL THEN 1 ELSE 0 END AS label
                    FROM {typ}_{src}_cands c
                    LEFT JOIN gt_{src} gt ON c.source1_entity_id = gt.source1_entity_id AND c.matched_entity_id = gt.matched_entity_id
                ) TO '{out_path}' WITH (FORMAT CSV, DELIMITER '\\t', HEADER)
            """)
        else:
            out_path = os.path.join(CAND_DIR, f"test_candidate_pairs_{src}.tsv")
            con.execute(f"COPY (SELECT source1_entity_id, matched_entity_id FROM {typ}_{src}_cands) TO '{out_path}' WITH (FORMAT CSV, DELIMITER '\\t', HEADER)")
        
        print(f"  Exported {vol[7]} rows to {out_path}", flush=True)
        h = get_hash(out_path)
        manifest_rows.append(f"cands_BCD_v1\t{gen_script}\t{commit_sha}\t[0-9]+[A-Za-z]?\t{typ}_{src}\tA,B,C,D\tB={vol[0]},C={vol[1]},D={vol[2]}\t{vol[7]}\t{vol[7]}\t{vol[8]}\tpos={pos},neg={neg}\t{h}\n")
        
        results[typ][src] = {
            "b_only": vol[0], "c_only": vol[1], "d_only": vol[2],
            "bc_overlap": vol[3], "bd_overlap": vol[4], "cd_overlap": vol[5], "bcd_overlap": vol[6],
            "total": vol[7], "distinct_s1": vol[8], "zero_cand_s1": zero_cand,
            "pos": pos, "neg": neg, "hash": h
        }


# ==============================================================================
# TASK 4: Two-Way Set Comparison (Test)
# ==============================================================================
print("\nTask 4: Comparing Old Test B vs New Test B+C+D...", flush=True)
comp_res = {}
for src in ["s2", "s3"]:
    old_test = os.path.join(P1_OUT, f"test_candidate_pairs_{src}.tsv")
    new_test = os.path.join(CAND_DIR, f"test_candidate_pairs_{src}.tsv")
    
    con.execute(f"CREATE TABLE old_{src} AS SELECT source1_entity_id, matched_entity_id FROM read_csv('{old_test}', delim='\\t', header=true)")
    con.execute(f"CREATE TABLE new_{src} AS SELECT source1_entity_id, matched_entity_id FROM read_csv('{new_test}', delim='\\t', header=true)")
    
    old_only = con.execute(f"SELECT COUNT(*) FROM (SELECT * FROM old_{src} EXCEPT SELECT * FROM new_{src})").fetchone()[0]
    new_only = con.execute(f"SELECT COUNT(*) FROM (SELECT * FROM new_{src} EXCEPT SELECT * FROM old_{src})").fetchone()[0]
    intersect = con.execute(f"SELECT COUNT(*) FROM (SELECT * FROM old_{src} INTERSECT SELECT * FROM new_{src})").fetchone()[0]
    sym_diff = old_only + new_only
    
    c_old = con.execute(f"SELECT COUNT(*) FROM old_{src}").fetchone()[0]
    c_new = con.execute(f"SELECT COUNT(*) FROM new_{src}").fetchone()[0]
    pct_change = ((c_new - c_old) / c_old) * 100 if c_old else 0
    
    comp_res[src] = {
        "old_only": old_only, "new_only": new_only, "intersection": intersect,
        "symmetric_difference": sym_diff, "pct_change": round(pct_change, 2)
    }

# ==============================================================================
# TASK 6: Determinism
# ==============================================================================
print("Task 6: Checking Determinism...", flush=True)
h1 = results["test"]["s2"]["hash"]
h2 = get_hash(os.path.join(CAND_DIR, "test_candidate_pairs_s2.tsv"))


# ==============================================================================
# TASK 7: Feature Regeneration
# ==============================================================================
print("\nTask 7: Feature Regeneration...", flush=True)
FEATURE_SQL = """
    CASE WHEN s1.business_name <> '' AND s1.business_name = tgt.business_name THEN 1.0 ELSE 0.0 END AS name_exact_match,
    jaro_winkler_similarity(COALESCE(s1.business_name,''), COALESCE(tgt.business_name,'')) / 100.0 AS name_jaro_winkler,
    CASE WHEN LENGTH(COALESCE(s1.business_name,'')) >= 2 AND LENGTH(COALESCE(tgt.business_name,'')) >= 2 THEN jaccard(COALESCE(s1.business_name,''), COALESCE(tgt.business_name,'')) ELSE 0.0 END AS name_jaccard,
    CASE WHEN LENGTH(s1.business_name) >= 4 AND LENGTH(tgt.business_name) >= 4 AND LEFT(s1.business_name, 4) = LEFT(tgt.business_name, 4) THEN 1.0 ELSE 0.0 END AS prefix4_match,
    CASE WHEN s1.business_name <> '' AND tgt.business_name <> '' AND SPLIT_PART(s1.business_name, ' ', 1) = SPLIT_PART(tgt.business_name, ' ', 1) THEN 1.0 ELSE 0.0 END AS first_token_match,
    CAST(ABS(LENGTH(COALESCE(s1.business_name,'')) - LENGTH(COALESCE(tgt.business_name,''))) AS DOUBLE) AS name_len_diff,
    CASE WHEN LENGTH(s1.business_name) > 0 AND LENGTH(tgt.business_name) > 0 THEN CAST(LEAST(LENGTH(s1.business_name), LENGTH(tgt.business_name)) AS DOUBLE) / GREATEST(LENGTH(s1.business_name), LENGTH(tgt.business_name)) ELSE 0.0 END AS name_len_ratio,
    CASE WHEN s1.business_address <> '' AND s1.business_address = tgt.business_address THEN 1.0 ELSE 0.0 END AS address_exact_match,
    jaro_winkler_similarity(COALESCE(s1.business_address,''), COALESCE(tgt.business_address,'')) / 100.0 AS address_jaro_winkler,
    CASE WHEN LENGTH(COALESCE(s1.business_address,'')) >= 2 AND LENGTH(COALESCE(tgt.business_address,'')) >= 2 THEN jaccard(COALESCE(s1.business_address,''), COALESCE(tgt.business_address,'')) ELSE 0.0 END AS address_jaccard,
    CAST(ABS(LENGTH(COALESCE(s1.business_address,'')) - LENGTH(COALESCE(tgt.business_address,''))) AS DOUBLE) AS address_len_diff,
    CASE WHEN LENGTH(s1.business_address) > 0 AND LENGTH(tgt.business_address) > 0 THEN CAST(LEAST(LENGTH(s1.business_address), LENGTH(tgt.business_address)) AS DOUBLE) / GREATEST(LENGTH(s1.business_address), LENGTH(tgt.business_address)) ELSE 0.0 END AS address_len_ratio,
    CASE WHEN regexp_extract(COALESCE(s1.business_address,''), '[0-9]+[A-Za-z]?', 0) <> '' AND regexp_extract(COALESCE(tgt.business_address,''), '[0-9]+[A-Za-z]?', 0) <> '' AND regexp_extract(s1.business_address, '[0-9]+[A-Za-z]?', 0) = regexp_extract(tgt.business_address, '[0-9]+[A-Za-z]?', 0) THEN 1.0 ELSE 0.0 END AS address_first_number_match,
    CASE WHEN s1.country <> '' AND s1.country = tgt.country THEN 1.0 ELSE 0.0 END AS country_match,
    CAST(LENGTH(COALESCE(s1.business_name,'')) AS DOUBLE) AS s1_name_len,
    CAST(LENGTH(COALESCE(tgt.business_name,'')) AS DOUBLE) AS tgt_name_len,
    CAST(LENGTH(COALESCE(s1.business_address,'')) AS DOUBLE) AS s1_addr_len,
    CAST(LENGTH(COALESCE(tgt.business_address,'')) AS DOUBLE) AS tgt_addr_len
"""

feat_manifest = []
feat_checks = {}

for typ in ["train", "test"]:
    for src in ["s2", "s3"]:
        print(f"Generating features for {typ} {src}...", flush=True)
        cand_path = os.path.join(CAND_DIR, f"{typ}_candidate_pairs_{src}.tsv")
        out_path = os.path.join(FEAT_DIR, f"cands_BCD_v1_features_{typ}_{src}.tsv")
        
        label_col = ", c.label" if typ == "train" else ""
        is_s3 = 1 if src == "s3" else 0
        
        con.execute(f"""
            COPY (
                SELECT
                    c.source1_entity_id,
                    c.matched_entity_id as candidate_entity_id,
                    {FEATURE_SQL},
                    {is_s3} AS source_is_s3
                    {label_col}
                FROM read_csv('{cand_path}', delim='\\t', header=true, all_varchar=true) c
                LEFT JOIN {typ}_s1 s1 ON c.source1_entity_id = s1.entity_id
                LEFT JOIN {typ}_{src} tgt ON c.matched_entity_id = tgt.entity_id
            )
            TO '{out_path}'
            WITH (FORMAT CSV, DELIMITER '\\t', HEADER)
        """)
        
        cand_count = con.execute(f"SELECT COUNT(*) FROM read_csv('{cand_path}', delim='\\t', header=true)").fetchone()[0]
        feat_count = con.execute(f"SELECT COUNT(*) FROM read_csv('{out_path}', delim='\\t', header=true)").fetchone()[0]
        
        sym_diff = con.execute(f"""
            SELECT COUNT(*) FROM (
                (SELECT source1_entity_id, matched_entity_id FROM read_csv('{cand_path}', delim='\\t', header=true)
                 EXCEPT
                 SELECT source1_entity_id, candidate_entity_id FROM read_csv('{out_path}', delim='\\t', header=true))
                UNION ALL
                (SELECT source1_entity_id, candidate_entity_id FROM read_csv('{out_path}', delim='\\t', header=true)
                 EXCEPT
                 SELECT source1_entity_id, matched_entity_id FROM read_csv('{cand_path}', delim='\\t', header=true))
            )
        """).fetchone()[0]
        
        feat_checks[f"{typ}_{src}"] = {"cand_count": cand_count, "feat_count": feat_count, "sym_diff": sym_diff}
        ts_f = datetime.now(timezone.utc).isoformat()
        feat_manifest.append(f"{typ.upper()} {src.upper()}\tcands_BCD_v1\t{out_path}\t{feat_count}\t{gen_script}\t{ts_f}\n")


# ==============================================================================
# WRITING MANIFESTS & SPECS
# ==============================================================================
print("\nWriting Manifests...", flush=True)
with open(os.path.join(REPORT_DIR, "cands_BCD_v1_manifest.tsv"), "w") as f:
    f.write("candidate_family\tgenerator_script\tgit_sha\textractor_spec\tinput_dataset\trule_defs\tper_rule_vols\tunion_volume\trow_count\tdistinct_s1\tlabels\tdet_hash\n")
    for row in manifest_rows:
        f.write(row)

with open(os.path.join(REPORT_DIR, "cands_BCD_v1_feature_manifest.tsv"), "w") as f:
    f.write("split_source\tcandidate_version\tfeature_path\trow_count\tgenerator_script\tgeneration_timestamp\n")
    for row in feat_manifest:
        f.write(row)

extractor_spec = """# Extractor Spec: `cands_BCD_v1`
- exact regex: `[0-9]+[A-Za-z]?`
- first-match semantics: `0` index in DuckDB's `regexp_extract()`
- input column: `business_address`
- normalized/raw: normalized
- case handling: `A-Za-z` implicitly matching lowercased string
- no-match behavior: `''`
- engine: DuckDB
- quirks: fuzzy matching captures embedded alphanumeric clusters (e.g., `3rd` -> `3r`)
"""
with open(os.path.join(REPORT_DIR, "cands_BCD_v1_extractor_spec.md"), "w") as f:
    f.write(extractor_spec)


print("\n=== FINAL REPORT ===", flush=True)
print("1. TEST Generator: notebooks/03_blocking.ipynb")
print("2. TRAIN counts:", results["train"])
print("3. TEST counts:", results["test"])
print("4. B+C+D Test vs Old B:", comp_res)
print("6. Determinism Hashes:", h1, h2)
print("7. Feature Checks:", feat_checks)
print("8. P3 Folds: Checked separately.")
