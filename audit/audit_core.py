#!/usr/bin/env python3
"""
audit/audit_core.py — Core forensic audit script.
Independently measures key pipeline metrics from actual data files.
Does NOT modify any production artifact.
"""
import os, sys, json, hashlib, time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)

import duckdb

TMP = os.path.join(REPO, "audit", "duckdb_tmp")
os.makedirs(TMP, exist_ok=True)
os.makedirs(os.path.join(REPO, "audit"), exist_ok=True)

con = duckdb.connect()
con.execute(f"SET temp_directory='{TMP}'")
con.execute("SET memory_limit='8GB'")
con.execute("SET threads=4")

results = {}

# ===== SECTION 1: Raw Data Row Counts =====
print("="*60)
print("SECTION 1: Raw Data Row Counts")
print("="*60)

raw_counts = {}
TRAIN_DIR = os.path.join(REPO, "data", "train")
TEST_DIR = os.path.join(REPO, "data", "test")
NORM_DIR = os.path.join(REPO, "outputs", "person1_step1", "normalized")

for name, path in [
    ("train_source1", os.path.join(TRAIN_DIR, "train_source1.tsv")),
    ("train_source2", os.path.join(TRAIN_DIR, "train_source2.tsv")),
    ("train_source3", os.path.join(TRAIN_DIR, "train_source3.tsv")),
    ("train_ground_truth", os.path.join(TRAIN_DIR, "train_ground_truth.tsv")),
    ("test_source1", os.path.join(TEST_DIR, "test_source1.tsv")),
    ("test_source2", os.path.join(TEST_DIR, "test_source2.tsv")),
    ("test_source3", os.path.join(TEST_DIR, "test_source3.tsv")),
]:
    if not os.path.exists(path):
        print(f"  {name}: FILE NOT FOUND")
        raw_counts[name] = "FILE NOT FOUND"
        continue
    cnt = con.execute(f"SELECT COUNT(*) FROM read_csv('{path}', delim='\\t', header=true, all_varchar=true)").fetchone()[0]
    cols = con.execute(f"SELECT * FROM read_csv('{path}', delim='\\t', header=true, all_varchar=true, sample_size=1)").description
    col_names = [c[0] for c in cols]
    print(f"  {name}: {cnt:,} rows, cols={col_names}")
    raw_counts[name] = {"rows": cnt, "columns": col_names}

results["raw_data_counts"] = raw_counts

# ===== SECTION 2: Normalized Data Row Counts =====
print("\n" + "="*60)
print("SECTION 2: Normalized Data Row Counts")
print("="*60)

norm_counts = {}
for name, path in [
    ("train_source1_normalized", os.path.join(NORM_DIR, "train_source1_normalized.tsv")),
    ("train_source2_normalized", os.path.join(NORM_DIR, "train_source2_normalized.tsv")),
    ("train_source3_normalized", os.path.join(NORM_DIR, "train_source3_normalized.tsv")),
    ("test_source1_normalized", os.path.join(NORM_DIR, "test_source1_normalized.tsv")),
    ("test_source2_normalized", os.path.join(NORM_DIR, "test_source2_normalized.tsv")),
    ("test_source3_normalized", os.path.join(NORM_DIR, "test_source3_normalized.tsv")),
]:
    if not os.path.exists(path):
        print(f"  {name}: FILE NOT FOUND")
        norm_counts[name] = "FILE NOT FOUND"
        continue
    cnt = con.execute(f"SELECT COUNT(*) FROM read_csv('{path}', delim='\\t', header=true, all_varchar=true)").fetchone()[0]
    cols = con.execute(f"SELECT * FROM read_csv('{path}', delim='\\t', header=true, all_varchar=true, sample_size=1)").description
    col_names = [c[0] for c in cols]
    uid = con.execute(f"SELECT COUNT(DISTINCT entity_id) FROM read_csv('{path}', delim='\\t', header=true, all_varchar=true)").fetchone()[0]
    print(f"  {name}: {cnt:,} rows, {uid:,} unique IDs, cols={col_names}")
    norm_counts[name] = {"rows": cnt, "unique_ids": uid, "columns": col_names}

results["normalized_data_counts"] = norm_counts

# ===== SECTION 3: Row Retention (Raw vs Normalized) =====
print("\n" + "="*60)
print("SECTION 3: Row Retention")
print("="*60)

retention = {}
for base in ["train_source1", "train_source2", "train_source3", "test_source1", "test_source2", "test_source3"]:
    raw = raw_counts.get(base, {})
    norm = norm_counts.get(f"{base}_normalized", {})
    if isinstance(raw, str) or isinstance(norm, str):
        print(f"  {base}: CANNOT COMPARE (missing file)")
        retention[base] = "MISSING"
        continue
    raw_rows = raw["rows"]
    norm_rows = norm["rows"]
    diff = raw_rows - norm_rows
    print(f"  {base}: raw={raw_rows:,}, norm={norm_rows:,}, diff={diff}")
    retention[base] = {"raw": raw_rows, "normalized": norm_rows, "rows_lost": diff}

results["row_retention"] = retention

# ===== SECTION 4: Ground Truth Forensic Audit =====
print("\n" + "="*60)
print("SECTION 4: Ground Truth Audit")
print("="*60)

gt_path = os.path.join(TRAIN_DIR, "train_ground_truth.tsv")
gt_recon_path = os.path.join(REPO, "outputs", "person1_step1", "train_ground_truth_reconstructed.tsv")

gt_metrics = {}

# Raw GT
cnt = con.execute(f"SELECT COUNT(*) FROM read_csv('{gt_path}', delim='\\t', header=true, all_varchar=true)").fetchone()[0]
uid = con.execute(f"SELECT COUNT(DISTINCT source1_entity_id) FROM read_csv('{gt_path}', delim='\\t', header=true, all_varchar=true)").fetchone()[0]
print(f"  Raw GT: {cnt:,} rows, {uid:,} unique S1 IDs")
gt_metrics["raw_gt_rows"] = cnt
gt_metrics["raw_gt_unique_s1"] = uid

# Reconstructed GT comparison
if os.path.exists(gt_recon_path):
    recon_cnt = con.execute(f"SELECT COUNT(*) FROM read_csv('{gt_recon_path}', delim='\\t', header=true, all_varchar=true)").fetchone()[0]
    recon_uid = con.execute(f"SELECT COUNT(DISTINCT source1_entity_id) FROM read_csv('{gt_recon_path}', delim='\\t', header=true, all_varchar=true)").fetchone()[0]
    print(f"  Reconstructed GT: {recon_cnt:,} rows, {recon_uid:,} unique S1 IDs")
    
    # SHA comparison
    gt_sha = hashlib.sha256(open(gt_path, 'rb').read()).hexdigest()
    recon_sha = hashlib.sha256(open(gt_recon_path, 'rb').read()).hexdigest()
    sha_match = gt_sha == recon_sha
    print(f"  SHA256 match: {sha_match}")
    print(f"    Raw GT SHA:   {gt_sha[:16]}...")
    print(f"    Recon GT SHA: {recon_sha[:16]}...")
    gt_metrics["recon_gt_rows"] = recon_cnt
    gt_metrics["recon_gt_unique_s1"] = recon_uid
    gt_metrics["sha_match"] = sha_match

# GT Exploded pair counts
con.execute(f"""
    CREATE OR REPLACE TABLE gt_exp AS 
    SELECT source1_entity_id, TRIM(UNNEST(string_split(matched_entity_ids, ','))) AS matched_entity_id
    FROM read_csv('{gt_path}', delim='\\t', header=true, all_varchar=true)
    WHERE matched_entity_ids IS NOT NULL AND TRIM(matched_entity_ids) <> ''
""")
total_pairs = con.execute("SELECT COUNT(*) FROM gt_exp").fetchone()[0]
s2_pairs = con.execute("SELECT COUNT(*) FROM gt_exp WHERE matched_entity_id LIKE 'S2-%'").fetchone()[0]
s3_pairs = con.execute("SELECT COUNT(*) FROM gt_exp WHERE matched_entity_id LIKE 'S3-%'").fetchone()[0]
other_pairs = total_pairs - s2_pairs - s3_pairs

# No-match S1
no_match_s1 = con.execute(f"""
    SELECT COUNT(*) FROM read_csv('{gt_path}', delim='\\t', header=true, all_varchar=true) 
    WHERE matched_entity_ids IS NULL OR TRIM(matched_entity_ids) = ''
""").fetchone()[0]

# Duplicate S1 check in GT
dup_s1_gt = con.execute(f"""
    SELECT COUNT(*) FROM (
        SELECT source1_entity_id, COUNT(*) as c FROM read_csv('{gt_path}', delim='\\t', header=true, all_varchar=true)
        GROUP BY source1_entity_id HAVING c > 1
    )
""").fetchone()[0]

print(f"  Total GT pairs (exploded): {total_pairs:,}")
print(f"    S2 pairs: {s2_pairs:,}")
print(f"    S3 pairs: {s3_pairs:,}")
print(f"    Other pairs: {other_pairs:,}")
print(f"  S1 with no match: {no_match_s1:,}")
print(f"  Duplicate S1 IDs in GT: {dup_s1_gt}")

gt_metrics["total_exploded_pairs"] = total_pairs
gt_metrics["s2_pairs"] = s2_pairs
gt_metrics["s3_pairs"] = s3_pairs
gt_metrics["other_pairs"] = other_pairs
gt_metrics["no_match_s1"] = no_match_s1
gt_metrics["dup_s1_in_gt"] = dup_s1_gt

# S1 match distribution
s1_match_counts = con.execute(f"""
    WITH per_s1 AS (
        SELECT source1_entity_id, 
               COUNT(CASE WHEN matched_entity_id LIKE 'S2-%' THEN 1 END) AS s2_cnt,
               COUNT(CASE WHEN matched_entity_id LIKE 'S3-%' THEN 1 END) AS s3_cnt,
               COUNT(*) as total_cnt
        FROM gt_exp
        GROUP BY source1_entity_id
    )
    SELECT
        COUNT(CASE WHEN s2_cnt > 0 AND s3_cnt > 0 THEN 1 END) AS both,
        COUNT(CASE WHEN s2_cnt > 0 AND s3_cnt = 0 THEN 1 END) AS s2_only,
        COUNT(CASE WHEN s2_cnt = 0 AND s3_cnt > 0 THEN 1 END) AS s3_only
    FROM per_s1
""").fetchone()
print(f"  S1 matched to both S2&S3: {s1_match_counts[0]:,}")
print(f"  S1 matched to S2 only: {s1_match_counts[1]:,}")
print(f"  S1 matched to S3 only: {s1_match_counts[2]:,}")

gt_metrics["s1_both_s2_s3"] = s1_match_counts[0]
gt_metrics["s1_s2_only"] = s1_match_counts[1]
gt_metrics["s1_s3_only"] = s1_match_counts[2]

results["ground_truth"] = gt_metrics

# ===== SECTION 5: Candidate Statistics (V1 - used by E02) =====
print("\n" + "="*60)
print("SECTION 5: Candidate File Statistics (cands_BCD_v1)")
print("="*60)

CAND_DIR = os.path.join(REPO, "P2", "data", "candidates")
cand_stats = {}

for name, path in [
    ("train_s2_v1", os.path.join(CAND_DIR, "train_candidate_pairs_s2.tsv")),
    ("train_s3_v1", os.path.join(CAND_DIR, "train_candidate_pairs_s3.tsv")),
    ("test_s2_p1", os.path.join(REPO, "outputs", "person1_step1", "test_candidate_pairs_s2.tsv")),
    ("test_s3_p1", os.path.join(REPO, "outputs", "person1_step1", "test_candidate_pairs_s3.tsv")),
    ("train_s2_v2", os.path.join(CAND_DIR, "train_candidate_pairs_s2_v2.tsv")),
    ("train_s3_v2", os.path.join(CAND_DIR, "train_candidate_pairs_s3_v2.tsv")),
    ("test_s2_v2", os.path.join(CAND_DIR, "test_candidate_pairs_s2_v2.tsv")),
    ("test_s3_v2", os.path.join(CAND_DIR, "test_candidate_pairs_s3_v2.tsv")),
]:
    if not os.path.exists(path):
        print(f"  {name}: FILE NOT FOUND")
        cand_stats[name] = "FILE NOT FOUND"
        continue
    cols = con.execute(f"SELECT * FROM read_csv('{path}', delim='\\t', header=true, all_varchar=true, sample_size=1)").description
    col_names = [c[0] for c in cols]
    cnt = con.execute(f"SELECT COUNT(*) FROM read_csv('{path}', delim='\\t', header=true, all_varchar=true)").fetchone()[0]
    uid_s1 = con.execute(f"SELECT COUNT(DISTINCT source1_entity_id) FROM read_csv('{path}', delim='\\t', header=true, all_varchar=true)").fetchone()[0]
    uid_tgt = con.execute(f"SELECT COUNT(DISTINCT matched_entity_id) FROM read_csv('{path}', delim='\\t', header=true, all_varchar=true)").fetchone()[0]
    
    # Label stats (if label column exists)
    label_info = {}
    if "label" in col_names:
        pos = con.execute(f"SELECT COUNT(*) FROM read_csv('{path}', delim='\\t', header=true) WHERE CAST(label AS INT) = 1").fetchone()[0]
        neg = cnt - pos
        label_info = {"positives": pos, "negatives": neg}
    
    print(f"  {name}: {cnt:,} pairs, {uid_s1:,} unique S1, {uid_tgt:,} unique target, cols={col_names}")
    if label_info:
        print(f"    Pos: {label_info['positives']:,}, Neg: {label_info['negatives']:,}")
    
    cand_stats[name] = {"rows": cnt, "unique_s1": uid_s1, "unique_target": uid_tgt, "columns": col_names, **label_info}

results["candidate_stats"] = cand_stats

# ===== SECTION 6: Candidate Recall (V1 — used by E02) =====
print("\n" + "="*60)
print("SECTION 6: Candidate Recall (cands_BCD_v1 = train used by E02)")
print("="*60)

recall_stats = {}
for src, cand_path_key, gt_src_filter in [
    ("S2", "train_s2_v1", "S2-%"),
    ("S3", "train_s3_v1", "S3-%"),
]:
    cand_path = os.path.join(CAND_DIR, f"train_candidate_pairs_{src.lower()}.tsv")
    if not os.path.exists(cand_path):
        print(f"  {src}: CANDIDATE FILE MISSING")
        recall_stats[src] = "MISSING"
        continue
    
    # Load candidates
    con.execute(f"""
        CREATE OR REPLACE TABLE cands_{src} AS 
        SELECT source1_entity_id, matched_entity_id 
        FROM read_csv('{cand_path}', delim='\\t', header=true, all_varchar=true)
    """)
    
    # Create GT for this source
    con.execute(f"""
        CREATE OR REPLACE TABLE gt_{src} AS 
        SELECT DISTINCT source1_entity_id, matched_entity_id 
        FROM gt_exp WHERE matched_entity_id LIKE '{gt_src_filter}'
    """)
    
    gt_total = con.execute(f"SELECT COUNT(*) FROM gt_{src}").fetchone()[0]
    
    # Captured = GT pairs present in candidates
    captured = con.execute(f"""
        SELECT COUNT(*) FROM gt_{src} gt
        WHERE EXISTS (
            SELECT 1 FROM cands_{src} c 
            WHERE c.source1_entity_id = gt.source1_entity_id 
            AND c.matched_entity_id = gt.matched_entity_id
        )
    """).fetchone()[0]
    
    missed = gt_total - captured
    recall = captured / gt_total if gt_total > 0 else 0
    
    # S1-level recall
    s1_with_gt = con.execute(f"SELECT COUNT(DISTINCT source1_entity_id) FROM gt_{src}").fetchone()[0]
    s1_fully_covered = con.execute(f"""
        WITH truth AS (
            SELECT source1_entity_id, COUNT(*) as tc FROM gt_{src} GROUP BY 1
        ),
        captured AS (
            SELECT gt.source1_entity_id, COUNT(*) as cc FROM gt_{src} gt
            JOIN cands_{src} c ON gt.source1_entity_id = c.source1_entity_id 
                AND gt.matched_entity_id = c.matched_entity_id
            GROUP BY 1
        )
        SELECT COUNT(*) FROM truth t JOIN captured c ON t.source1_entity_id = c.source1_entity_id
        WHERE t.tc = c.cc
    """).fetchone()[0]
    
    s1_partial = con.execute(f"""
        WITH captured AS (
            SELECT DISTINCT gt.source1_entity_id FROM gt_{src} gt
            JOIN cands_{src} c ON gt.source1_entity_id = c.source1_entity_id 
                AND gt.matched_entity_id = c.matched_entity_id
        )
        SELECT COUNT(*) FROM captured
    """).fetchone()[0] - s1_fully_covered
    
    s1_zero_capture = s1_with_gt - s1_fully_covered - s1_partial
    
    print(f"  {src}: GT total={gt_total:,}, captured={captured:,}, missed={missed:,}, recall={recall:.6f}")
    print(f"    S1 with GT: {s1_with_gt:,}")
    print(f"    S1 fully covered: {s1_fully_covered:,}")
    print(f"    S1 partial: {s1_partial:,}")
    print(f"    S1 zero capture: {s1_zero_capture:,}")
    
    recall_stats[src] = {
        "gt_total": gt_total,
        "captured": captured,
        "missed": missed,
        "recall": round(recall, 6),
        "s1_with_gt": s1_with_gt,
        "s1_fully_covered": s1_fully_covered,
        "s1_partial": s1_partial,
        "s1_zero_capture": s1_zero_capture,
    }

results["candidate_recall_v1"] = recall_stats

# ===== SECTION 7: Candidate Recall (V2) =====
print("\n" + "="*60)
print("SECTION 7: Candidate Recall (cands_BCD_v2)")
print("="*60)

recall_v2 = {}
for src in ["S2", "S3"]:
    cand_path = os.path.join(CAND_DIR, f"train_candidate_pairs_{src.lower()}_v2.tsv")
    if not os.path.exists(cand_path):
        print(f"  {src}: V2 CANDIDATE FILE MISSING")
        recall_v2[src] = "MISSING"
        continue
    
    con.execute(f"""
        CREATE OR REPLACE TABLE candsv2_{src} AS 
        SELECT source1_entity_id, matched_entity_id 
        FROM read_csv('{cand_path}', delim='\\t', header=true, all_varchar=true)
    """)
    
    gt_total = con.execute(f"SELECT COUNT(*) FROM gt_{src}").fetchone()[0]
    captured = con.execute(f"""
        SELECT COUNT(*) FROM gt_{src} gt
        WHERE EXISTS (
            SELECT 1 FROM candsv2_{src} c 
            WHERE c.source1_entity_id = gt.source1_entity_id 
            AND c.matched_entity_id = gt.matched_entity_id
        )
    """).fetchone()[0]
    
    missed = gt_total - captured
    recall = captured / gt_total if gt_total > 0 else 0
    
    print(f"  {src}: GT={gt_total:,}, captured={captured:,}, missed={missed:,}, recall={recall:.6f}")
    recall_v2[src] = {
        "gt_total": gt_total, "captured": captured, "missed": missed, "recall": round(recall, 6)
    }

results["candidate_recall_v2"] = recall_v2

# ===== SECTION 8: Pipeline Disconnection Audit =====
print("\n" + "="*60)
print("SECTION 8: Pipeline Disconnection Audit")
print("="*60)

# Check which candidates the actual pipeline uses (phase3_to_8 / e02_train_lgb)
print("  phase3_to_8_pipeline.py references:")
print("    TRAIN_S2_CAND = P2/data/candidates/train_strategy_b_candidates_s2.tsv")
print("    TRAIN_S3_CAND = P2/data/candidates/train_strategy_b_candidates_s3.tsv")
print(f"    train_strategy_b_candidates_s2.tsv exists: {os.path.exists(os.path.join(CAND_DIR, 'train_strategy_b_candidates_s2.tsv'))}")
print(f"    train_strategy_b_candidates_s3.tsv exists: {os.path.exists(os.path.join(CAND_DIR, 'train_strategy_b_candidates_s3.tsv'))}")

print("  e02_train_lgb.py references:")
print("    FEAT_DIR/cands_BCD_v1_features_train_s2.tsv")
print("    FEAT_DIR/cands_BCD_v1_features_train_s3.tsv")
feat_s2_exists = os.path.exists(os.path.join(REPO, "P2", "data", "features", "cands_BCD_v1_features_train_s2.tsv"))
feat_s3_exists = os.path.exists(os.path.join(REPO, "P2", "data", "features", "cands_BCD_v1_features_train_s3.tsv"))
print(f"    cands_BCD_v1_features_train_s2.tsv exists: {feat_s2_exists}")
print(f"    cands_BCD_v1_features_train_s3.tsv exists: {feat_s3_exists}")
print(f"    Model files exist: {os.path.exists(os.path.join(REPO, 'P2', 'models'))}")
model_dir = os.path.join(REPO, "P2", "models")
if os.path.exists(model_dir):
    models = os.listdir(model_dir)
    print(f"    Models in dir: {models}")
else:
    print(f"    Models directory does not exist")

# ===== SECTION 9: Normalization Column Check =====
print("\n" + "="*60)
print("SECTION 9: Normalization Column Audit")
print("="*60)

# Check what columns exist in normalized files
for name, path in [
    ("train_source1_norm", os.path.join(NORM_DIR, "train_source1_normalized.tsv")),
]:
    cols = con.execute(f"SELECT * FROM read_csv('{path}', delim='\\t', header=true, all_varchar=true, sample_size=1)").description
    col_names = [c[0] for c in cols]
    print(f"  {name} columns: {col_names}")
    
    # Check how many empty business_name / business_address  
    empty_name = con.execute(f"SELECT COUNT(*) FROM read_csv('{path}', delim='\\t', header=true, all_varchar=true) WHERE business_name IS NULL OR TRIM(business_name) = ''").fetchone()[0]
    empty_addr = con.execute(f"SELECT COUNT(*) FROM read_csv('{path}', delim='\\t', header=true, all_varchar=true) WHERE business_address IS NULL OR TRIM(business_address) = ''").fetchone()[0]
    empty_country = con.execute(f"SELECT COUNT(*) FROM read_csv('{path}', delim='\\t', header=true, all_varchar=true) WHERE country IS NULL OR TRIM(country) = ''").fetchone()[0]
    total = con.execute(f"SELECT COUNT(*) FROM read_csv('{path}', delim='\\t', header=true, all_varchar=true)").fetchone()[0]
    print(f"    Empty/null business_name: {empty_name:,} / {total:,}")
    print(f"    Empty/null business_address: {empty_addr:,} / {total:,}")
    print(f"    Empty/null country: {empty_country:,} / {total:,}")

# ===== SECTION 10: Test Data Stats =====
print("\n" + "="*60)
print("SECTION 10: Test Data Stats")
print("="*60)

test_counts = {}
for name, path in [
    ("test_source1", os.path.join(TEST_DIR, "test_source1.tsv")),
    ("test_source2", os.path.join(TEST_DIR, "test_source2.tsv")),
    ("test_source3", os.path.join(TEST_DIR, "test_source3.tsv")),
]:
    cnt = con.execute(f"SELECT COUNT(*) FROM read_csv('{path}', delim='\\t', header=true, all_varchar=true)").fetchone()[0]
    uid = con.execute(f"SELECT COUNT(DISTINCT entity_id) FROM read_csv('{path}', delim='\\t', header=true, all_varchar=true)").fetchone()[0]
    print(f"  {name}: {cnt:,} rows, {uid:,} unique IDs")
    test_counts[name] = {"rows": cnt, "unique_ids": uid}

results["test_data_counts"] = test_counts

# ===== SECTION 11: E02 Metrics Cross-Check =====
print("\n" + "="*60)
print("SECTION 11: E02 Metrics Cross-Check")
print("="*60)

e02_path = os.path.join(REPO, "P2", "reports", "E02_RUN.json")
if os.path.exists(e02_path):
    with open(e02_path) as f:
        e02 = json.load(f)
    print(f"  Reported macro_f05: {e02.get('macro_f05')}")
    print(f"  Reported candidate_oracle: {e02.get('candidate_oracle')}")
    print(f"  Reported candidate_floor: {e02.get('candidate_floor')}")
    print(f"  Reported oof_row_count: {e02.get('oof_row_count')}")
    print(f"  Fixed threshold: {e02.get('fixed_threshold')}")
    print(f"  S2 rows: {e02.get('integrity_checks', {}).get('s2_rows')}")
    print(f"  S3 rows: {e02.get('integrity_checks', {}).get('s3_rows')}")
    
    # Cross-check: OOF row count should equal V1 train S2 + V1 train S3
    v1_s2 = cand_stats.get("train_s2_v1", {})
    v1_s3 = cand_stats.get("train_s3_v1", {})
    if isinstance(v1_s2, dict) and isinstance(v1_s3, dict):
        expected_oof = v1_s2.get("rows", 0) + v1_s3.get("rows", 0)
        actual_oof = e02.get("oof_row_count", 0)
        print(f"\n  CROSS-CHECK: V1 train S2 ({v1_s2.get('rows', 0):,}) + V1 train S3 ({v1_s3.get('rows', 0):,}) = {expected_oof:,}")
        print(f"  E02 reports oof_row_count = {actual_oof:,}")
        print(f"  Match: {expected_oof == actual_oof}")

# ===== SECTION 12: Candidate Recall to Oracle Gap =====
print("\n" + "="*60)
print("SECTION 12: Candidate Recall to Oracle Gap Analysis")
print("="*60)

# Compute theoretical oracle score from V1 candidates
# Oracle = for each S1, predict exactly the GT matches that are IN the candidate set
# Floor = for each S1, predict empty
# This needs the scorer
print("  E02 Reported:")
print(f"    Floor (empty prediction): {e02.get('candidate_floor', 'N/A')}")
print(f"    Baseline (T=0.5):        {e02.get('macro_f05', 'N/A')}")
print(f"    Oracle (perfect on cands): {e02.get('candidate_oracle', 'N/A')}")
print(f"    Model gap (oracle - baseline): {(e02.get('candidate_oracle', 0) - e02.get('macro_f05', 0)):.4f}")
print(f"    Candidate gap (1.0 - oracle):  {(1.0 - e02.get('candidate_oracle', 0)):.4f}")

results["e02_metrics"] = {
    "floor": e02.get("candidate_floor"),
    "baseline": e02.get("macro_f05"),
    "oracle": e02.get("candidate_oracle"),
    "model_gap": round(e02.get("candidate_oracle", 0) - e02.get("macro_f05", 0), 4),
    "candidate_gap": round(1.0 - e02.get("candidate_oracle", 0), 4),
}

# ===== SAVE =====
out_path = os.path.join(REPO, "audit", "audit_core_results.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\n\nResults saved to {out_path}")
print("AUDIT CORE COMPLETE")

con.close()
