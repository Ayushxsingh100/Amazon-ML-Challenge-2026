import os
import sys
import hashlib
import duckdb

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

MATCHING_T090 = os.path.join(REPO, "output", "t090", "matching_results.tsv")
MATCHING_T050 = os.path.join(REPO, "output", "matching_results.tsv")
CANDIDATE_FILE = os.path.join(REPO, "output", "candidate_pairs.tsv")
TEST_S1_FILE = os.path.join(REPO, "outputs", "person1_step1", "normalized", "test_source1_normalized.tsv")

def sha256_file(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()

def verify():
    con = duckdb.connect()
    
    # 1. SHA256 hashes
    print("=" * 70)
    print("SHA256 CHECKSUMS")
    print("=" * 70)
    sha_matching_t090 = sha256_file(MATCHING_T090)
    sha_candidate = sha256_file(CANDIDATE_FILE)
    sha_matching_t050 = sha256_file(MATCHING_T050)
    print(f"output/t090/matching_results.tsv : {sha_matching_t090}")
    print(f"output/candidate_pairs.tsv       : {sha_candidate}")
    print(f"output/matching_results.tsv      : {sha_matching_t050}")
    
    # 2. Test S1 population
    con.execute(f"""
        CREATE TABLE canonical_s1 AS
        SELECT entity_id AS source1_entity_id
        FROM read_csv('{TEST_S1_FILE.replace(os.sep, "/")}', delim='\\t', header=true, all_varchar=true)
    """)
    n_canonical_s1 = con.execute("SELECT COUNT(*) FROM canonical_s1").fetchone()[0]
    print(f"\nCanonical test S1 population: {n_canonical_s1:,}")
    
    # 3. Analyze T=0.90 matching file
    print("\n" + "=" * 70)
    print("ANALYZING T=0.90 MATCHING FILE (output/t090/matching_results.tsv)")
    print("=" * 70)
    
    con.execute(f"""
        CREATE TABLE match_t090 AS
        SELECT source1_entity_id, matched_entity_ids
        FROM read_csv('{MATCHING_T090.replace(os.sep, "/")}', delim='\\t', header=true, all_varchar=true)
    """)
    
    total_rows = con.execute("SELECT COUNT(*) FROM match_t090").fetchone()[0]
    distinct_s1 = con.execute("SELECT COUNT(DISTINCT source1_entity_id) FROM match_t090").fetchone()[0]
    duplicate_s1 = total_rows - distinct_s1
    
    non_empty_s1 = con.execute("SELECT COUNT(*) FROM match_t090 WHERE matched_entity_ids IS NOT NULL AND matched_entity_ids != ''").fetchone()[0]
    empty_s1 = con.execute("SELECT COUNT(*) FROM match_t090 WHERE matched_entity_ids IS NULL OR matched_entity_ids = ''").fetchone()[0]
    
    missing_s1 = con.execute("""
        SELECT COUNT(*) FROM canonical_s1 c
        LEFT JOIN match_t090 m ON c.source1_entity_id = m.source1_entity_id
        WHERE m.source1_entity_id IS NULL
    """).fetchone()[0]
    
    extra_s1 = con.execute("""
        SELECT COUNT(*) FROM match_t090 m
        LEFT JOIN canonical_s1 c ON m.source1_entity_id = c.source1_entity_id
        WHERE c.source1_entity_id IS NULL
    """).fetchone()[0]
    
    # Unnest match pairs
    con.execute("""
        CREATE TABLE unnested_pairs_t090 AS
        SELECT source1_entity_id, unnest(string_split(matched_entity_ids, ',')) AS matched_entity_id
        FROM match_t090
        WHERE matched_entity_ids IS NOT NULL AND matched_entity_ids != ''
    """)
    
    total_predicted_matches = con.execute("SELECT COUNT(*) FROM unnested_pairs_t090").fetchone()[0]
    distinct_predicted_matches = con.execute("SELECT COUNT(DISTINCT (source1_entity_id || '__' || matched_entity_id)) FROM unnested_pairs_t090").fetchone()[0]
    duplicate_match_ids = total_predicted_matches - distinct_predicted_matches
    
    invalid_ids = con.execute("""
        SELECT COUNT(*) FROM unnested_pairs_t090
        WHERE NOT (matched_entity_id LIKE 'S2-%' OR matched_entity_id LIKE 'S3-%')
    """).fetchone()[0]
    
    print(f"Total matching S1 rows         : {total_rows:,}")
    print(f"Distinct S1 rows               : {distinct_s1:,}")
    print(f"Duplicate S1 rows              : {duplicate_s1:,}")
    print(f"Non-empty S1 count             : {non_empty_s1:,}")
    print(f"Empty S1 count                 : {empty_s1:,}")
    print(f"Missing S1 from canonical      : {missing_s1:,}")
    print(f"Extra S1 not in canonical      : {extra_s1:,}")
    print(f"All S1 coverage complete       : {missing_s1 == 0 and extra_s1 == 0 and total_rows == 1732544}")
    print(f"Total predicted match pairs    : {total_predicted_matches:,}")
    print(f"Duplicate match IDs in an S1   : {duplicate_match_ids:,}")
    print(f"Invalid S2/S3 IDs              : {invalid_ids:,}")
    
    # 4. Check candidate universe membership (Task 5)
    print("\n" + "=" * 70)
    print("TASK 5 — CANDIDATE MEMBERSHIP CHECK")
    print("=" * 70)
    
    con.execute(f"""
        CREATE TABLE cand_pairs AS
        SELECT source1_entity_id, unnest(string_split(candidate_entity_ids, ',')) AS candidate_entity_id
        FROM read_csv('{CANDIDATE_FILE.replace(os.sep, "/")}', delim='\\t', header=true, all_varchar=true)
        WHERE candidate_entity_ids IS NOT NULL AND candidate_entity_ids != ''
    """)
    
    cand_count = con.execute("SELECT COUNT(*) FROM cand_pairs").fetchone()[0]
    print(f"Total candidate pairs in output/candidate_pairs.tsv : {cand_count:,}")
    
    non_cand_matches = con.execute("""
        SELECT COUNT(*)
        FROM unnested_pairs_t090 m
        LEFT JOIN cand_pairs c
          ON m.source1_entity_id = c.source1_entity_id
         AND m.matched_entity_id = c.candidate_entity_id
        WHERE c.candidate_entity_id IS NULL
    """).fetchone()[0]
    
    print(f"Non-candidate matches : {non_cand_matches}")
    
    # 5. Analyze T=0.50 matching file for comparison (Task 9)
    print("\n" + "=" * 70)
    print("TASK 9 — COMPARISON WITH T=0.50 FALLBACK")
    print("=" * 70)
    con.execute(f"""
        CREATE TABLE match_t050 AS
        SELECT source1_entity_id, matched_entity_ids
        FROM read_csv('{MATCHING_T050.replace(os.sep, "/")}', delim='\\t', header=true, all_varchar=true)
    """)
    t050_total = con.execute("SELECT COUNT(*) FROM match_t050").fetchone()[0]
    t050_non_empty = con.execute("SELECT COUNT(*) FROM match_t050 WHERE matched_entity_ids IS NOT NULL AND matched_entity_ids != ''").fetchone()[0]
    t050_empty = con.execute("SELECT COUNT(*) FROM match_t050 WHERE matched_entity_ids IS NULL OR matched_entity_ids = ''").fetchone()[0]
    
    con.execute("""
        CREATE TABLE unnested_pairs_t050 AS
        SELECT source1_entity_id, unnest(string_split(matched_entity_ids, ',')) AS matched_entity_id
        FROM match_t050
        WHERE matched_entity_ids IS NOT NULL AND matched_entity_ids != ''
    """)
    t050_matches = con.execute("SELECT COUNT(*) FROM unnested_pairs_t050").fetchone()[0]
    
    print("Metric                      | T=0.50 (Fallback) | T=0.90 (Experiment) | Delta")
    print("-" * 75)
    print(f"Predicted matches           | {t050_matches:>17,} | {total_predicted_matches:>19,} | {total_predicted_matches - t050_matches:>+10,}")
    print(f"Non-empty S1                | {t050_non_empty:>17,} | {non_empty_s1:>19,} | {non_empty_s1 - t050_non_empty:>+10,}")
    print(f"Empty S1                    | {t050_empty:>17,} | {empty_s1:>19,} | {empty_s1 - t050_empty:>+10,}")
    print(f"Total S1 rows               | {t050_total:>17,} | {total_rows:>19,} | {total_rows - t050_total:>+10}")

if __name__ == "__main__":
    verify()
