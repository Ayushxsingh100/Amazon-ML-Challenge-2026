#!/usr/bin/env python3
"""
Submission Validator for Amazon ML Challenge 2026.
Validates:
1. Matching results file structure and constraints
2. Candidate pairs file structure and constraints
3. Alignment between matching results and candidate pairs
4. Completeness against test Source 1 entities
5. ID format correctness (S1, S2, S3)
6. Absence of duplicate IDs and rows
7. Threshold / subset integrity
"""

import argparse
import os
import sys
import duckdb


def validate(matching_path, candidate_path, test_dir):
    print("=" * 70)
    print("SUBMISSION VALIDATOR -- AMAZON ML CHALLENGE 2026")
    print(f"Matching file:  {matching_path}")
    print(f"Candidate file: {candidate_path}")
    print(f"Test directory: {test_dir}")
    print("=" * 70)

    errors = []
    warnings = []

    if not os.path.exists(matching_path):
        print(f"FATAL: Matching file not found: {matching_path}")
        return False

    if not os.path.exists(candidate_path):
        print(f"FATAL: Candidate file not found: {candidate_path}")
        return False

    con = duckdb.connect()
    con.execute("SET memory_limit='6GB'")
    con.execute("SET threads=4")

    # 1. Determine canonical test S1 entities
    # Try normalized first, else scan test_dir
    norm_s1 = os.path.join("outputs", "person1_step1", "normalized", "test_source1_normalized.tsv")
    if os.path.exists(norm_s1):
        norm_s1_clean = norm_s1.replace("\\", "/")
        con.execute(f"CREATE TABLE canonical_s1 AS SELECT entity_id FROM read_csv('{norm_s1_clean}', delim='\\t', header=true, all_varchar=true)")
    else:
        # scan test directory parts
        s1_parts = os.path.join(test_dir, "test_source1.tsv.part_*").replace("\\", "/")
        con.execute(f"CREATE TABLE canonical_s1 AS SELECT entity_id FROM read_csv('{s1_parts}', delim='\\t', header=true, all_varchar=true)")

    expected_s1_count = con.execute("SELECT COUNT(*) FROM canonical_s1").fetchone()[0]
    print(f"Canonical Test Source 1 entities: {expected_s1_count:,}")

    # 2. Check Candidate Pairs File
    cand_p = candidate_path.replace("\\", "/")
    con.execute(f"CREATE TABLE cands_raw AS SELECT * FROM read_csv('{cand_p}', delim='\\t', header=true, all_varchar=true)")
    cand_cols = [col[0] for col in con.execute("DESCRIBE cands_raw").fetchall()]
    print(f"Candidate file columns: {cand_cols}")
    if len(cand_cols) != 2 or cand_cols[0] != "source1_entity_id":
        errors.append(f"Candidate file must have 2 columns starting with 'source1_entity_id', found {cand_cols}")

    cand_val_col = cand_cols[1] if len(cand_cols) >= 2 else "candidate_entity_ids"
    cand_s1_count = con.execute("SELECT COUNT(*) FROM cands_raw").fetchone()[0]
    cand_distinct_s1 = con.execute("SELECT COUNT(DISTINCT source1_entity_id) FROM cands_raw").fetchone()[0]

    print(f"Candidate file rows: {cand_s1_count:,} (Distinct S1: {cand_distinct_s1:,})")
    if cand_s1_count != expected_s1_count:
        errors.append(f"Candidate file row count ({cand_s1_count:,}) != expected test S1 count ({expected_s1_count:,})")
    if cand_distinct_s1 != cand_s1_count:
        errors.append(f"Candidate file contains duplicate S1 rows ({cand_s1_count - cand_distinct_s1} duplicates)")

    # 3. Check Matching Results File
    match_p = matching_path.replace("\\", "/")
    con.execute(f"CREATE TABLE matches_raw AS SELECT * FROM read_csv('{match_p}', delim='\\t', header=true, all_varchar=true)")
    match_cols = [col[0] for col in con.execute("DESCRIBE matches_raw").fetchall()]
    print(f"Matching file columns: {match_cols}")
    if len(match_cols) != 2 or match_cols[0] != "source1_entity_id":
        errors.append(f"Matching file must have 2 columns starting with 'source1_entity_id', found {match_cols}")

    match_val_col = match_cols[1] if len(match_cols) >= 2 else "matched_entity_ids"
    match_s1_count = con.execute("SELECT COUNT(*) FROM matches_raw").fetchone()[0]
    match_distinct_s1 = con.execute("SELECT COUNT(DISTINCT source1_entity_id) FROM matches_raw").fetchone()[0]

    print(f"Matching file rows: {match_s1_count:,} (Distinct S1: {match_distinct_s1:,})")
    if match_s1_count != expected_s1_count:
        errors.append(f"Matching file row count ({match_s1_count:,}) != expected test S1 count ({expected_s1_count:,})")
    if match_distinct_s1 != match_s1_count:
        errors.append(f"Matching file contains duplicate S1 rows ({match_s1_count - match_distinct_s1} duplicates)")

    # 4. Check S1 set equality
    missing_s1_cands = con.execute("""
        SELECT COUNT(*) FROM canonical_s1 c
        LEFT JOIN cands_raw r ON c.entity_id = r.source1_entity_id
        WHERE r.source1_entity_id IS NULL
    """).fetchone()[0]
    if missing_s1_cands > 0:
        errors.append(f"{missing_s1_cands:,} canonical S1 entities missing from candidate file")

    missing_s1_match = con.execute("""
        SELECT COUNT(*) FROM canonical_s1 c
        LEFT JOIN matches_raw r ON c.entity_id = r.source1_entity_id
        WHERE r.source1_entity_id IS NULL
    """).fetchone()[0]
    if missing_s1_match > 0:
        errors.append(f"{missing_s1_match:,} canonical S1 entities missing from matching file")

    # 5. Expand and validate individual IDs
    print("\nExpanding candidate pairs and matching pairs...")
    con.execute(f"""
        CREATE TABLE cands_exploded AS
        SELECT source1_entity_id,
               TRIM(UNNEST(string_split(COALESCE("{cand_val_col}", ''), ','))) AS cand_id
        FROM cands_raw
        WHERE "{cand_val_col}" IS NOT NULL AND TRIM("{cand_val_col}") <> ''
    """)
    total_cand_pairs = con.execute("SELECT COUNT(*) FROM cands_exploded").fetchone()[0]
    print(f"Total candidate pairs: {total_cand_pairs:,}")

    con.execute(f"""
        CREATE TABLE matches_exploded AS
        SELECT source1_entity_id,
               TRIM(UNNEST(string_split(COALESCE("{match_val_col}", ''), ','))) AS match_id
        FROM matches_raw
        WHERE "{match_val_col}" IS NOT NULL AND TRIM("{match_val_col}") <> ''
    """)
    total_match_pairs = con.execute("SELECT COUNT(*) FROM matches_exploded").fetchone()[0]
    print(f"Total predicted match pairs: {total_match_pairs:,}")

    # Check candidate ID prefixes (must be S2 or S3)
    invalid_cand_ids = con.execute("""
        SELECT COUNT(*) FROM cands_exploded
        WHERE NOT (cand_id LIKE 'S2-%' OR cand_id LIKE 'S3-%')
    """).fetchone()[0]
    if invalid_cand_ids > 0:
        errors.append(f"Found {invalid_cand_ids:,} candidate IDs with invalid prefix (not S2 or S3)")

    # Check match ID prefixes (must be S2 or S3)
    invalid_match_ids = con.execute("""
        SELECT COUNT(*) FROM matches_exploded
        WHERE NOT (match_id LIKE 'S2-%' OR match_id LIKE 'S3-%')
    """).fetchone()[0]
    if invalid_match_ids > 0:
        errors.append(f"Found {invalid_match_ids:,} match IDs with invalid prefix (not S2 or S3)")

    # Check for duplicate candidates within S1
    dup_cand_pairs = con.execute("""
        SELECT COUNT(*) FROM (
            SELECT source1_entity_id, cand_id, COUNT(*) AS cnt
            FROM cands_exploded
            GROUP BY source1_entity_id, cand_id
            HAVING COUNT(*) > 1
        )
    """).fetchone()[0]
    if dup_cand_pairs > 0:
        errors.append(f"Found {dup_cand_pairs:,} duplicate candidate pairs within S1 entities")

    # Check for duplicate matches within S1
    dup_match_pairs = con.execute("""
        SELECT COUNT(*) FROM (
            SELECT source1_entity_id, match_id, COUNT(*) AS cnt
            FROM matches_exploded
            GROUP BY source1_entity_id, match_id
            HAVING COUNT(*) > 1
        )
    """).fetchone()[0]
    if dup_match_pairs > 0:
        errors.append(f"Found {dup_match_pairs:,} duplicate match pairs within S1 entities")

    # 6. Check that every match is in candidate pairs
    non_cand_matches = con.execute("""
        SELECT COUNT(*) FROM matches_exploded m
        LEFT JOIN cands_exploded c
          ON m.source1_entity_id = c.source1_entity_id
         AND m.match_id = c.cand_id
        WHERE c.cand_id IS NULL
    """).fetchone()[0]
    if non_cand_matches > 0:
        errors.append(f"Found {non_cand_matches:,} matches that are NOT in the candidate pairs file!")

    # 7. Statistics
    empty_matches_count = con.execute(f"""
        SELECT COUNT(*) FROM matches_raw
        WHERE "{match_val_col}" IS NULL OR TRIM("{match_val_col}") = ''
    """).fetchone()[0]
    active_matches_s1 = expected_s1_count - empty_matches_count

    empty_cands_count = con.execute(f"""
        SELECT COUNT(*) FROM cands_raw
        WHERE "{cand_val_col}" IS NULL OR TRIM("{cand_val_col}") = ''
    """).fetchone()[0]
    active_cands_s1 = expected_s1_count - empty_cands_count

    print("\n--- Summary Statistics ---")
    print(f"Total S1 entities:                  {expected_s1_count:,}")
    print(f"S1 with at least 1 candidate:       {active_cands_s1:,} ({active_cands_s1/expected_s1_count*100:.2f}%)")
    print(f"S1 with 0 candidates:               {empty_cands_count:,} ({empty_cands_count/expected_s1_count*100:.2f}%)")
    print(f"Total candidate pairs:              {total_cand_pairs:,}")
    print(f"S1 with at least 1 match:           {active_matches_s1:,} ({active_matches_s1/expected_s1_count*100:.2f}%)")
    print(f"S1 with 0 matches:                  {empty_matches_count:,} ({empty_matches_count/expected_s1_count*100:.2f}%)")
    print(f"Total predicted matches:            {total_match_pairs:,}")

    con.close()

    print("\n" + "=" * 70)
    if errors:
        print("VALIDATION RESULT: FAIL")
        print("Errors found:")
        for e in errors:
            print(f"  [ERROR] {e}")
        return False
    else:
        print("VALIDATION RESULT: PASS")
        if warnings:
            for w in warnings:
                print(f"  [WARNING] {w}")
        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate submission files")
    parser.add_argument("--matching", required=True, help="Path to matching_results.tsv")
    parser.add_argument("--candidate", required=True, help="Path to candidate_pairs.tsv")
    parser.add_argument("--test-dir", required=True, help="Path to test data directory")
    args = parser.parse_args()

    success = validate(args.matching, args.candidate, args.test_dir)
    sys.exit(0 if success else 1)
