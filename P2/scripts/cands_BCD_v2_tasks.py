#!/usr/bin/env python3
"""
P2/scripts/cands_BCD_v2_tasks.py
Authoritative candidate generation pipeline v2 for Amazon ML Challenge 2026.
Builds upon canonical cands_BCD_v1 by incorporating:
  - Clean alphanumeric business name matching (Rule E)
  - Zero-stripped house number normalization for prefix4 & token1 (Rule F)
  - Semantic root-token matching with house normalization (Rule G)
  - Clean alphanumeric address matching (Rule H)
  - High-precision postal code + name prefix3 matching (Rule I)

Output candidate files:
  - P2/data/candidates/train_candidate_pairs_s2_v2.tsv
  - P2/data/candidates/train_candidate_pairs_s3_v2.tsv
  - P2/data/candidates/test_candidate_pairs_s2_v2.tsv
  - P2/data/candidates/test_candidate_pairs_s3_v2.tsv
"""

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
TMP_DIR = os.path.join(P2_DIR, "data", "duckdb_tmp_v2_gen")

os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(CAND_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)

con = duckdb.connect()
con.execute(f"SET temp_directory='{TMP_DIR}'")
con.execute("SET memory_limit='8GB'")
con.execute("SET threads=4")

COMMON_PREFIXES = "('the', 'shri', 'sri', 'dr', 'm/s', 'hotel', 'new', 'om', 'sai', 'jai', 'a', 'an')"

print("1. Loading normalized data with V2 blocking keys...", flush=True)
for typ in ["train", "test"]:
    for src in ["1", "2", "3"]:
        path = os.path.join(NORM_DIR, f"{typ}_source{src}_normalized.tsv")
        con.execute(f"""
            CREATE OR REPLACE TABLE {typ}_s{src} AS 
            SELECT entity_id, business_name, business_address, country,
                   LEFT(TRIM(business_name), 4) AS prefix4,
                   split_part(TRIM(business_name), ' ', 1) AS token1,
                   regexp_extract(TRIM(business_address), '[0-9]+[A-Za-z]?', 0) AS house_v1,
                   regexp_replace(regexp_extract(TRIM(business_address), '[0-9]+[A-Za-z]?', 0), '^0+', '') AS house_norm,
                   regexp_replace(lower(trim(business_name)), '[^a-z0-9]', '', 'g') AS clean_name,
                   regexp_replace(lower(trim(business_address)), '[^a-z0-9]', '', 'g') AS clean_addr,
                   LEFT(TRIM(business_name), 3) AS prefix3,
                   CASE 
                       WHEN lower(split_part(TRIM(business_name), ' ', 1)) IN {COMMON_PREFIXES} 
                            AND split_part(TRIM(business_name), ' ', 2) <> '' 
                       THEN split_part(TRIM(business_name), ' ', 2)
                       ELSE split_part(TRIM(business_name), ' ', 1)
                   END AS root_token,
                   regexp_extract(TRIM(business_address), '(?:^|[^0-9])([0-9]{{5,6}})(?:[^0-9]|$)', 1) AS zip_code
            FROM read_csv('{path}', delim='\\t', header=true, all_varchar=true)
        """)

print("2. Loading training ground truth...", flush=True)
gt_path = os.path.join(P1_OUT, "train_ground_truth_reconstructed.tsv")
con.execute(f"""
    CREATE OR REPLACE TABLE gt_exp AS 
    SELECT source1_entity_id, TRIM(UNNEST(string_split(matched_entity_ids, ','))) AS matched_entity_id
    FROM read_csv('{gt_path}', delim='\\t', header=true, all_varchar=true)
    WHERE matched_entity_ids IS NOT NULL AND TRIM(matched_entity_ids) <> ''
""")
con.execute("CREATE OR REPLACE TABLE gt_s2 AS SELECT * FROM gt_exp WHERE matched_entity_id LIKE 'S2-%'")
con.execute("CREATE OR REPLACE TABLE gt_s3 AS SELECT * FROM gt_exp WHERE matched_entity_id LIKE 'S3-%'")

def get_file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def get_sorted_pairs_sha256(path):
    res = con.execute(f"SELECT source1_entity_id || '-' || matched_entity_id FROM read_csv('{path}', delim='\\t', header=true) ORDER BY 1").fetchall()
    h = hashlib.sha256()
    for row in res:
        h.update(row[0].encode("utf-8"))
    return h.hexdigest()

results = {}
manifest_rows = []
commit_sha = os.popen("git rev-parse HEAD").read().strip()
gen_script = "P2/scripts/cands_BCD_v2_tasks.py"
ts = datetime.now(timezone.utc).isoformat()

print("\n3. Generating candidate pairs for V2 (Train & Test)...", flush=True)

for typ in ["train", "test"]:
    results[typ] = {}
    for src in ["s2", "s3"]:
        t0 = time.time()
        print(f"\n--- Generating {typ} {src} V2 Candidates ---", flush=True)
        
        # Candidate Generation Query combining V1 baseline rules + selected V2 rules
        con.execute(f"""
            CREATE OR REPLACE TABLE {typ}_{src}_v2_raw AS
            -- Rule A: country + prefix4 + house_v1
            SELECT s1.entity_id as source1_entity_id, tgt.entity_id as matched_entity_id
            FROM {typ}_s1 s1 JOIN {typ}_{src} tgt 
            ON s1.country <> '' AND s1.country = tgt.country 
               AND s1.prefix4 <> '' AND s1.prefix4 = tgt.prefix4 
               AND s1.house_v1 <> '' AND s1.house_v1 = tgt.house_v1
            
            UNION
            
            -- Rule B: country + exact normalized business_address
            SELECT s1.entity_id, tgt.entity_id
            FROM {typ}_s1 s1 JOIN {typ}_{src} tgt 
            ON s1.country <> '' AND s1.country = tgt.country 
               AND s1.business_address <> '' AND s1.business_address = tgt.business_address
            
            UNION
            
            -- Rule C: country + token1 + house_v1
            SELECT s1.entity_id, tgt.entity_id
            FROM {typ}_s1 s1 JOIN {typ}_{src} tgt 
            ON s1.country <> '' AND s1.country = tgt.country 
               AND s1.token1 <> '' AND s1.token1 = tgt.token1 
               AND s1.house_v1 <> '' AND s1.house_v1 = tgt.house_v1
            
            UNION
            
            -- Rule D: country + exact normalized business_name
            SELECT s1.entity_id, tgt.entity_id
            FROM {typ}_s1 s1 JOIN {typ}_{src} tgt 
            ON s1.country <> '' AND s1.country = tgt.country 
               AND TRIM(s1.business_name) <> '' AND TRIM(s1.business_name) = TRIM(tgt.business_name)
            
            UNION
            
            -- Rule E: country + clean alphanumeric business name
            SELECT s1.entity_id, tgt.entity_id
            FROM {typ}_s1 s1 JOIN {typ}_{src} tgt 
            ON s1.country <> '' AND s1.country = tgt.country 
               AND s1.clean_name <> '' AND LENGTH(s1.clean_name) >= 3 AND s1.clean_name = tgt.clean_name
               
            UNION
            
            -- Rule F1: country + prefix4 + house_norm (leading zeros stripped)
            SELECT s1.entity_id, tgt.entity_id
            FROM {typ}_s1 s1 JOIN {typ}_{src} tgt 
            ON s1.country <> '' AND s1.country = tgt.country 
               AND s1.prefix4 <> '' AND s1.prefix4 = tgt.prefix4 
               AND s1.house_norm <> '' AND s1.house_norm = tgt.house_norm
               
            UNION
            
            -- Rule F2: country + token1 + house_norm (leading zeros stripped)
            SELECT s1.entity_id, tgt.entity_id
            FROM {typ}_s1 s1 JOIN {typ}_{src} tgt 
            ON s1.country <> '' AND s1.country = tgt.country 
               AND s1.token1 <> '' AND s1.token1 = tgt.token1 
               AND s1.house_norm <> '' AND s1.house_norm = tgt.house_norm
               
            UNION
            
            -- Rule G: country + root_token + house_norm
            SELECT s1.entity_id, tgt.entity_id
            FROM {typ}_s1 s1 JOIN {typ}_{src} tgt 
            ON s1.country <> '' AND s1.country = tgt.country 
               AND s1.root_token <> '' AND LENGTH(s1.root_token) >= 3 AND s1.root_token = tgt.root_token 
               AND s1.house_norm <> '' AND s1.house_norm = tgt.house_norm
               
            UNION
            
            -- Rule H: country + clean alphanumeric address
            SELECT s1.entity_id, tgt.entity_id
            FROM {typ}_s1 s1 JOIN {typ}_{src} tgt 
            ON s1.country <> '' AND s1.country = tgt.country 
               AND s1.clean_addr <> '' AND LENGTH(s1.clean_addr) >= 6 AND s1.clean_addr = tgt.clean_addr
               
            UNION
            
            -- Rule I: country + postal_code + prefix3
            SELECT s1.entity_id, tgt.entity_id
            FROM {typ}_s1 s1 JOIN {typ}_{src} tgt 
            ON s1.country <> '' AND s1.country = tgt.country 
               AND s1.zip_code <> '' AND LENGTH(s1.zip_code) >= 5 AND s1.zip_code = tgt.zip_code 
               AND s1.prefix3 <> '' AND LENGTH(s1.prefix3) = 3 AND s1.prefix3 = tgt.prefix3
        """)
        
        cand_count = con.execute(f"SELECT COUNT(*) FROM {typ}_{src}_v2_raw").fetchone()[0]
        distinct_s1 = con.execute(f"SELECT COUNT(DISTINCT source1_entity_id) FROM {typ}_{src}_v2_raw").fetchone()[0]
        print(f"Generated {cand_count:,} unique candidate pairs ({distinct_s1:,} distinct S1 entities) in {time.time()-t0:.1f}s", flush=True)
        
        pos = neg = 0
        if typ == "train":
            out_path = os.path.join(CAND_DIR, f"train_candidate_pairs_{src}_v2.tsv")
            con.execute(f"""
                COPY (
                    SELECT c.source1_entity_id, c.matched_entity_id,
                           CASE WHEN gt.matched_entity_id IS NOT NULL THEN 1 ELSE 0 END AS label
                    FROM {typ}_{src}_v2_raw c
                    LEFT JOIN gt_{src} gt ON c.source1_entity_id = gt.source1_entity_id AND c.matched_entity_id = gt.matched_entity_id
                ) TO '{out_path}' WITH (FORMAT CSV, DELIMITER '\\t', HEADER)
            """)
            
            pos = con.execute(f"SELECT COUNT(*) FROM read_csv('{out_path}', delim='\\t', header=true) WHERE CAST(label AS INT) = 1").fetchone()[0]
            neg = cand_count - pos
            gt_total = con.execute(f"SELECT COUNT(*) FROM gt_{src}").fetchone()[0]
            recall = pos / gt_total
            print(f"Exported {cand_count:,} rows to {out_path} | Pos={pos:,} | Neg={neg:,} | Recall={recall*100:.3f}%", flush=True)
        else:
            out_path = os.path.join(CAND_DIR, f"test_candidate_pairs_{src}_v2.tsv")
            con.execute(f"""
                COPY (
                    SELECT source1_entity_id, matched_entity_id 
                    FROM {typ}_{src}_v2_raw
                ) TO '{out_path}' WITH (FORMAT CSV, DELIMITER '\\t', HEADER)
            """)
            print(f"Exported {cand_count:,} rows to {out_path}", flush=True)
        
        file_sha = get_file_sha256(out_path)
        pair_sha = get_sorted_pairs_sha256(out_path)
        file_size = os.path.getsize(out_path)
        
        manifest_rows.append({
            "candidate_family": "cands_BCD_v2",
            "generator_script": gen_script,
            "git_sha": commit_sha,
            "input_dataset": f"{typ}_{src}",
            "rule_defs": "A,B,C,D,E(CleanName),F(HouseNorm),G(RootToken),H(CleanAddr),I(ZipPrefix3)",
            "row_count": cand_count,
            "distinct_s1": distinct_s1,
            "positives": pos,
            "negatives": neg,
            "file_size_bytes": file_size,
            "file_sha256": file_sha,
            "sorted_pairs_sha256": pair_sha,
            "path": out_path
        })
        
        results[typ][src] = {
            "total_candidates": cand_count,
            "distinct_s1": distinct_s1,
            "positives": pos,
            "negatives": neg,
            "file_sha256": file_sha,
            "sorted_pairs_sha256": pair_sha,
            "file_size": file_size,
            "out_path": out_path
        }

# Write Manifest
manifest_path = os.path.join(REPORT_DIR, "cands_BCD_v2_manifest.tsv")
with open(manifest_path, "w") as f:
    headers = ["candidate_family", "generator_script", "git_sha", "input_dataset", "rule_defs", "row_count", "distinct_s1", "positives", "negatives", "file_size_bytes", "file_sha256", "sorted_pairs_sha256", "path"]
    f.write("\t".join(headers) + "\n")
    for r in manifest_rows:
        f.write("\t".join(str(r[h]) for h in headers) + "\n")

print(f"\nSaved Manifest to {manifest_path}")

# Write Extractor Spec
extractor_spec = """# Extractor Spec: `cands_BCD_v2`

## Pipeline Architecture
`cands_BCD_v2` is an authoritative, measured extension of canonical `cands_BCD_v1`. It retains all existing v1 baseline candidate generation rules while resolving the primary root causes of missed pairs identified via anti-join analysis.

## Blocking Key Rules
1. **Rule A (V1)**: Country + `prefix4(business_name)` + `house_v1`
2. **Rule B (V1)**: Country + exact normalized `business_address`
3. **Rule C (V1)**: Country + `token1(business_name)` + `house_v1`
4. **Rule D (V1)**: Country + exact normalized `business_name`
5. **Rule E (Clean Name)**: Country + alphanumeric clean name (`regexp_replace(lower(business_name), '[^a-z0-9]', '', 'g')`), min length >= 3.
   - *Rationale*: Captures punctuation differences ('inc.' vs 'inc', hyphens, brackets, special corporate annotations).
6. **Rule F (House Normalization)**:
   - `house_norm` = `regexp_replace(regexp_extract(business_address, '[0-9]+[A-Za-z]?', 0), '^0+', '')` (stripping leading zeros: e.g. `00622` -> `622`).
   - Combined with `prefix4` and `token1`.
7. **Rule G (Semantic Root Token + Normalized House)**:
   - `root_token`: Strips common non-discriminative prefixes ('the', 'shri', 'sri', 'dr', 'm/s', 'hotel', 'new', 'om', 'sai', 'jai') and extracts the true semantic entity root.
   - Combined with `house_norm`.
8. **Rule H (Clean Address)**:
   - Country + alphanumeric clean address (`regexp_replace(lower(business_address), '[^a-z0-9]', '', 'g')`), min length >= 6.
   - Precision proxy: > 81%.
9. **Rule I (Postal Code + Prefix3)**:
   - Country + 5-to-6 digit postal code (`[0-9]{5,6}`) + `prefix3(business_name)`.
   - Precision proxy: > 94.9%.

## Bounded Blocking and Safety
- Uncontrolled token-pair blocking (e.g. `token1 + token2` without bounds) was measured and explicitly **rejected** due to Cartesian explosion on generic multi-token names ('physical therapy', 'pediatric dental').
- Uncontrolled `prefix3 + house` without root-token filtering was measured and **rejected** due to 15M low-precision candidate explosion.
- All included V2 rules demonstrated verified marginal recall gain with strictly bounded candidate multipliers (total volume < 1.24x of V1).
"""
with open(os.path.join(REPORT_DIR, "cands_BCD_v2_extractor_spec.md"), "w") as f:
    f.write(extractor_spec)

print("Saved extractor spec to cands_BCD_v2_extractor_spec.md")
print("\n=== V2 GENERATION COMPLETE ===")
