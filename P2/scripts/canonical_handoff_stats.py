import os
import duckdb
import hashlib
import json
import time
from datetime import datetime, timezone

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
NORM_DIR = os.path.join(REPO, "outputs", "person1_step1", "normalized")
P2_DIR = os.path.join(REPO, "P2")
REPORT_DIR = os.path.join(P2_DIR, "reports")
FEAT_DIR = os.path.join(P2_DIR, "data", "features")
CAND_DIR = os.path.join(P2_DIR, "data", "candidates")
DB_TMP = os.path.join(P2_DIR, "data", "duckdb_tmp_v2")

os.makedirs(DB_TMP, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(FEAT_DIR, exist_ok=True)

con = duckdb.connect()
con.execute(f"SET temp_directory='{DB_TMP}'")
con.execute("SET memory_limit='8GB'")
con.execute("SET threads=4")

# ==============================================================================
# TASK 1: Correct V2 Label
# ==============================================================================
print("Task 1: Correcting V2 label...")
val_json_path = os.path.join(REPORT_DIR, "strategy_b_canonical_validation.json")
if os.path.exists(val_json_path):
    with open(val_json_path, "r", encoding="utf-8") as f:
        val_data = json.load(f)
    if "V2" in val_data:
        # Clarify the exact meaning of V2 differences
        val_data["V2_clarification"] = (
            "The 760,671 difference in V2 represents the count of differing rows between "
            "the OLD P2 extractor (\\b word boundaries) and the CANONICAL P1-style extractor ([0-9]+[A-Za-z]?). "
            "It is NOT a disagreement between the corrected P2 implementation and P1."
        )
    with open(val_json_path, "w", encoding="utf-8") as f:
        json.dump(val_data, f, indent=2)

# ==============================================================================
# Generate the train candidates using cands_B_v1 logic first
# ==============================================================================
print("Regenerating train candidates with canonical logic...")
os.system(f'python "{os.path.join(P2_DIR, "scripts", "generate_train_candidates.py")}"')

# ==============================================================================
# TASK 4: Train Candidate Counts
# ==============================================================================
print("Task 4: Calculating Train Candidate Counts...")
s2_cand_path = os.path.join(CAND_DIR, "train_candidate_pairs_s2.tsv")
s3_cand_path = os.path.join(CAND_DIR, "train_candidate_pairs_s3.tsv")

con.execute(f"CREATE TABLE s2_cands AS SELECT * FROM read_csv('{s2_cand_path}', delim='\\t', header=true)")
con.execute(f"CREATE TABLE s3_cands AS SELECT * FROM read_csv('{s3_cand_path}', delim='\\t', header=true)")

counts = {}
for tgt, tbl in [("S2", "s2_cands"), ("S3", "s3_cands")]:
    res = con.execute(f"""
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN CAST(label AS INT)=1 THEN 1 ELSE 0 END) as pos,
            SUM(CASE WHEN CAST(label AS INT)=0 THEN 1 ELSE 0 END) as neg
        FROM {tbl}
    """).fetchone()
    counts[tgt] = {"total": res[0], "positives": res[1], "negatives": res[2]}

# ==============================================================================
# TASK 5: Entity Coverage
# ==============================================================================
print("Task 5: Entity Coverage...")
s1_path = os.path.join(NORM_DIR, "train_source1_normalized.tsv")
con.execute(f"CREATE TABLE s1_ent AS SELECT entity_id FROM read_csv('{s1_path}', delim='\\t', header=true)")

gt_path = os.path.join(REPO, "outputs", "person1_step1", "train_ground_truth_reconstructed.tsv")
con.execute(f"""
    CREATE TABLE gt_exp AS 
    SELECT source1_entity_id, TRIM(UNNEST(string_split(matched_entity_ids, ','))) AS matched_entity_id
    FROM read_csv('{gt_path}', delim='\\t', header=true)
    WHERE matched_entity_ids IS NOT NULL AND matched_entity_ids <> ''
""")

coverage = {}
for tgt, tbl in [("S2", "s2_cands"), ("S3", "s3_cands")]:
    con.execute(f"CREATE TABLE gt_{tgt} AS SELECT * FROM gt_exp WHERE matched_entity_id LIKE '{tgt}-%'")
    
    # Total S1 entities
    total_s1 = con.execute("SELECT COUNT(*) FROM s1_ent").fetchone()[0]
    
    # S1 with zero candidates
    zero_cand = con.execute(f"""
        SELECT COUNT(*) FROM s1_ent 
        WHERE entity_id NOT IN (SELECT source1_entity_id FROM {tbl})
    """).fetchone()[0]
    
    # S1 with at least one candidate
    at_least_one_cand = total_s1 - zero_cand
    
    # S1 with truth but zero captured
    truth_zero_cap = con.execute(f"""
        SELECT COUNT(DISTINCT gt.source1_entity_id) FROM gt_{tgt} gt
        WHERE gt.source1_entity_id NOT IN (
            SELECT source1_entity_id FROM {tbl} WHERE CAST(label AS INT) = 1
        )
    """).fetchone()[0]
    
    # S1 with no truth
    no_truth = con.execute(f"""
        SELECT COUNT(*) FROM s1_ent 
        WHERE entity_id NOT IN (SELECT source1_entity_id FROM gt_{tgt})
    """).fetchone()[0]
    
    # Fully covered S1 entities (all truth pairs captured)
    # Total truth pairs for an entity == Captured truth pairs for that entity
    fully_cov = con.execute(f"""
        WITH TruthCounts AS (
            SELECT source1_entity_id, COUNT(*) as tc FROM gt_{tgt} GROUP BY 1
        ),
        CapturedCounts AS (
            SELECT source1_entity_id, COUNT(*) as cc FROM {tbl} WHERE CAST(label AS INT) = 1 GROUP BY 1
        )
        SELECT COUNT(*) FROM TruthCounts t
        LEFT JOIN CapturedCounts c ON t.source1_entity_id = c.source1_entity_id
        WHERE t.tc = coalesce(c.cc, 0)
    """).fetchone()[0]
    
    coverage[tgt] = {
        "total_s1_entities": total_s1,
        "s1_zero_candidates": zero_cand,
        "s1_at_least_one_candidate": at_least_one_cand,
        "s1_truth_but_zero_captured": truth_zero_cap,
        "s1_no_truth": no_truth,
        "s1_fully_covered": fully_cov
    }

# ==============================================================================
# TASK 3: Manifests
# ==============================================================================
print("Task 3: Creating Manifest...")
def get_file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def get_sorted_pairs_hash(path):
    res = con.execute(f"SELECT source1_entity_id || '-' || matched_entity_id FROM read_csv('{path}', delim='\\t', header=true) ORDER BY 1").fetchall()
    h = hashlib.sha256()
    for row in res:
        h.update(row[0].encode("utf-8"))
    return h.hexdigest()

manifest_path = os.path.join(REPORT_DIR, "cands_B_v1_manifest.tsv")
script_path = "P2/scripts/generate_train_candidates.py"
commit_sha = "9dfb66980c169f215abc20f3e85bfedcb5908be6"
ts = datetime.now(timezone.utc).isoformat()

with open(manifest_path, "w", encoding="utf-8") as f:
    f.write("candidate_version\tsource\tcandidate_path\trow_count\tsha256_sorted_pairs\tgenerator_script\tgenerator_commit\tinput_file_hashes\tgeneration_timestamp\n")
    
    # Train S2
    f.write(f"cands_B_v1\tTRAIN S2\t{s2_cand_path}\t{counts['S2']['total']}\t{get_sorted_pairs_hash(s2_cand_path)}\t{script_path}\t{commit_sha}\t-\t{ts}\n")
    # Train S3
    f.write(f"cands_B_v1\tTRAIN S3\t{s3_cand_path}\t{counts['S3']['total']}\t{get_sorted_pairs_hash(s3_cand_path)}\t{script_path}\t{commit_sha}\t-\t{ts}\n")
    
    # For Test, we use P1's exact files (since they are identical)
    test_s2 = os.path.join(REPO, "outputs", "person1_step1", "test_candidate_pairs_s2.tsv")
    test_s3 = os.path.join(REPO, "outputs", "person1_step1", "test_candidate_pairs_s3.tsv")
    f.write(f"cands_B_v1\tTEST S2\t{test_s2}\t26046195\t{get_sorted_pairs_hash(test_s2)}\t{script_path}\t{commit_sha}\t-\t{ts}\n")
    f.write(f"cands_B_v1\tTEST S3\t{test_s3}\t30777878\t{get_sorted_pairs_hash(test_s3)}\t{script_path}\t{commit_sha}\t-\t{ts}\n")


# Output final metrics for user
print("\n=== FINAL METRICS ===")
print("Candidate Counts:")
print(json.dumps(counts, indent=2))
print("Entity Coverage:")
print(json.dumps(coverage, indent=2))
