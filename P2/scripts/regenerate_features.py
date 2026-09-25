import os
import duckdb
import hashlib
import time
from datetime import datetime, timezone

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
NORM_DIR = os.path.join(REPO, "outputs", "person1_step1", "normalized")
P2_DIR = os.path.join(REPO, "P2")
REPORT_DIR = os.path.join(P2_DIR, "reports")
FEAT_DIR = os.path.join(P2_DIR, "data", "features")
CAND_DIR = os.path.join(P2_DIR, "data", "candidates")
DB_TMP = os.path.join(P2_DIR, "data", "duckdb_tmp_feats")

os.makedirs(DB_TMP, exist_ok=True)
os.makedirs(FEAT_DIR, exist_ok=True)

con = duckdb.connect()
con.execute(f"SET temp_directory='{DB_TMP}'")
con.execute("SET memory_limit='8GB'")
con.execute("SET threads=4")

FEATURE_SQL = """
    -- Name features
    CASE WHEN s1.business_name <> '' AND s1.business_name = tgt.business_name THEN 1.0 ELSE 0.0 END AS name_exact_match,
    jaro_winkler_similarity(COALESCE(s1.business_name,''), COALESCE(tgt.business_name,'')) / 100.0 AS name_jaro_winkler,
    CASE WHEN LENGTH(COALESCE(s1.business_name,'')) >= 2 AND LENGTH(COALESCE(tgt.business_name,'')) >= 2 THEN jaccard(COALESCE(s1.business_name,''), COALESCE(tgt.business_name,'')) ELSE 0.0 END AS name_jaccard,
    CASE WHEN LENGTH(s1.business_name) >= 4 AND LENGTH(tgt.business_name) >= 4
              AND LEFT(s1.business_name, 4) = LEFT(tgt.business_name, 4) THEN 1.0 ELSE 0.0 END AS prefix4_match,
    CASE WHEN s1.business_name <> '' AND tgt.business_name <> ''
              AND SPLIT_PART(s1.business_name, ' ', 1) = SPLIT_PART(tgt.business_name, ' ', 1) THEN 1.0 ELSE 0.0 END AS first_token_match,
    CAST(ABS(LENGTH(COALESCE(s1.business_name,'')) - LENGTH(COALESCE(tgt.business_name,''))) AS DOUBLE) AS name_len_diff,
    CASE WHEN LENGTH(s1.business_name) > 0 AND LENGTH(tgt.business_name) > 0
         THEN CAST(LEAST(LENGTH(s1.business_name), LENGTH(tgt.business_name)) AS DOUBLE) / GREATEST(LENGTH(s1.business_name), LENGTH(tgt.business_name))
         ELSE 0.0 END AS name_len_ratio,
    -- Address features
    CASE WHEN s1.business_address <> '' AND s1.business_address = tgt.business_address THEN 1.0 ELSE 0.0 END AS address_exact_match,
    jaro_winkler_similarity(COALESCE(s1.business_address,''), COALESCE(tgt.business_address,'')) / 100.0 AS address_jaro_winkler,
    CASE WHEN LENGTH(COALESCE(s1.business_address,'')) >= 2 AND LENGTH(COALESCE(tgt.business_address,'')) >= 2 THEN jaccard(COALESCE(s1.business_address,''), COALESCE(tgt.business_address,'')) ELSE 0.0 END AS address_jaccard,
    CAST(ABS(LENGTH(COALESCE(s1.business_address,'')) - LENGTH(COALESCE(tgt.business_address,''))) AS DOUBLE) AS address_len_diff,
    CASE WHEN LENGTH(s1.business_address) > 0 AND LENGTH(tgt.business_address) > 0
         THEN CAST(LEAST(LENGTH(s1.business_address), LENGTH(tgt.business_address)) AS DOUBLE) / GREATEST(LENGTH(s1.business_address), LENGTH(tgt.business_address))
         ELSE 0.0 END AS address_len_ratio,
    -- House number overlap (Canonical Extractor)
    CASE WHEN regexp_extract(COALESCE(s1.business_address,''), '[0-9]+[A-Za-z]?', 0) <> ''
              AND regexp_extract(COALESCE(tgt.business_address,''), '[0-9]+[A-Za-z]?', 0) <> ''
              AND regexp_extract(s1.business_address, '[0-9]+[A-Za-z]?', 0) = regexp_extract(tgt.business_address, '[0-9]+[A-Za-z]?', 0)
         THEN 1.0 ELSE 0.0 END AS address_first_number_match,
    -- Country
    CASE WHEN s1.country <> '' AND s1.country = tgt.country THEN 1.0 ELSE 0.0 END AS country_match,
    -- Lengths for model context
    CAST(LENGTH(COALESCE(s1.business_name,'')) AS DOUBLE) AS s1_name_len,
    CAST(LENGTH(COALESCE(tgt.business_name,'')) AS DOUBLE) AS tgt_name_len,
    CAST(LENGTH(COALESCE(s1.business_address,'')) AS DOUBLE) AS s1_addr_len,
    CAST(LENGTH(COALESCE(tgt.business_address,'')) AS DOUBLE) AS tgt_addr_len
"""

# Load normalized tables
print("Loading normalized data...")
for typ in ["train", "test"]:
    for src in ["s1", "s2", "s3"]:
        tbl = f"{typ}_{src}"
        num = src[-1]
        path = os.path.join(NORM_DIR, f"{typ}_source{num}_normalized.tsv")
        con.execute(f"CREATE TABLE {tbl} AS SELECT entity_id, business_name, business_address, country FROM read_csv('{path}', delim='\\t', header=true, all_varchar=true)")

tasks = [
    {"type": "train", "src": "s2", "cand": os.path.join(CAND_DIR, "train_candidate_pairs_s2.tsv"), "has_label": True},
    {"type": "train", "src": "s3", "cand": os.path.join(CAND_DIR, "train_candidate_pairs_s3.tsv"), "has_label": True},
    {"type": "test", "src": "s2", "cand": os.path.join(REPO, "outputs", "person1_step1", "test_candidate_pairs_s2.tsv"), "has_label": False},
    {"type": "test", "src": "s3", "cand": os.path.join(REPO, "outputs", "person1_step1", "test_candidate_pairs_s3.tsv"), "has_label": False},
]

manifest_rows = []

for task in tasks:
    typ = task["type"]
    src = task["src"]
    out_name = f"cands_B_v1_features_{typ}_{src}.tsv"
    out_path = os.path.join(FEAT_DIR, out_name)
    
    print(f"Generating features for {typ} {src}...")
    t0 = time.time()
    
    label_col = ", c.label" if task["has_label"] else ""
    is_s3 = 1 if src == "s3" else 0
    
    query = f"""
        COPY (
            SELECT
                c.source1_entity_id,
                c.matched_entity_id as candidate_entity_id,
                {FEATURE_SQL},
                {is_s3} AS source_is_s3
                {label_col}
            FROM read_csv('{task["cand"]}', delim='\\t', header=true, all_varchar=true) c
            LEFT JOIN {typ}_s1 s1 ON c.source1_entity_id = s1.entity_id
            LEFT JOIN {typ}_{src} tgt ON c.matched_entity_id = tgt.entity_id
        )
        TO '{out_path}'
        WITH (FORMAT CSV, DELIMITER '\\t', HEADER)
    """
    con.execute(query)
    
    # Verify exact row counts
    cand_count = con.execute(f"SELECT COUNT(*) FROM read_csv('{task['cand']}', delim='\\t', header=true)").fetchone()[0]
    feat_count = con.execute(f"SELECT COUNT(*) FROM read_csv('{out_path}', delim='\\t', header=true)").fetchone()[0]
    
    print(f"  Rows: cand={cand_count}, feat={feat_count}")
    if cand_count != feat_count:
        print("  WARNING: ROW COUNT MISMATCH!")
    
    ts = datetime.now(timezone.utc).isoformat()
    manifest_rows.append(f"{typ.upper()} {src.upper()}\tcands_B_v1\t{out_path}\t{feat_count}\tP2/scripts/regenerate_features.py\t{ts}\n")

with open(os.path.join(REPORT_DIR, "cands_B_v1_feature_manifest.tsv"), "w") as f:
    f.write("split_source\tcandidate_version\tfeature_path\trow_count\tgenerator_script\tgeneration_timestamp\n")
    for row in manifest_rows:
        f.write(row)
        
print("Feature regeneration complete!")
