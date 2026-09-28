#!/usr/bin/env python3
"""
Amazon ML Challenge 2026 - Phase 5 Final Matching & Submission Generator
========================================================================

Deterministic generator that produces the final test submission from validated
Phase 4 predictions.

Submission format:
- TSV with tab delimiter (\t)
- Header: source1_entity_id\tmatched_entity_ids
- Exactly one row per test S1 entity (1,732,544 data rows)
- matched_entity_ids: comma-separated target entity IDs (S2 and S3 combined)
- No-match: empty string after the tab (no quotes, no placeholder tokens)
- Deterministic ordering: source1_entity_id ASC, candidate ordering by model_score DESC, candidate_entity_id ASC

Inputs:
- P2/predictions/phase4/test_predictions_s2.tsv
- P2/predictions/phase4/test_predictions_s3.tsv
- P1/data/entities/test/source1/test_s1_entities.parquet

Outputs:
- P2/predictions/phase5/matching_results.tsv
- output/matching_results.tsv
"""

import os
import sys
import time
import hashlib
import duckdb

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Inputs
S2_PRED_PATH = os.path.join(REPO_ROOT, "P2", "predictions", "phase4", "test_predictions_s2.tsv")
S3_PRED_PATH = os.path.join(REPO_ROOT, "P2", "predictions", "phase4", "test_predictions_s3.tsv")
TEST_S1_PARQUET = os.path.join(REPO_ROOT, "P1", "data", "entities", "test", "source1", "test_s1_entities.parquet")
TEST_S2_PARQUET = os.path.join(REPO_ROOT, "P1", "data", "entities", "test", "source2", "test_s2_entities.parquet")
TEST_S3_PARQUET = os.path.join(REPO_ROOT, "P1", "data", "entities", "test", "source3", "test_s3_entities.parquet")

# Outputs
PHASE5_DIR = os.path.join(REPO_ROOT, "P2", "predictions", "phase5")
OUTPUT_DIR = os.path.join(REPO_ROOT, "output")
SUBMISSION_P5_PATH = os.path.join(PHASE5_DIR, "matching_results.tsv")
SUBMISSION_ROOT_PATH = os.path.join(OUTPUT_DIR, "matching_results.tsv")

# Expected constants
EXPECTED_TEST_S1_COUNT = 1732544
EXPECTED_S2_ROWS = 43841928
EXPECTED_S3_ROWS = 51354667
EXPECTED_S2_SHA256 = "1a4e94cc9b15ba984588db2b763e6c15967b2d7a7ef04213833d79dfe11307af"
EXPECTED_S3_SHA256 = "503282181cff0df65b86f3f19c2f2813b4cd84d99a1a2b4ed167ac2199d5e3aa"
THRESHOLD = 0.60


def compute_sha256(filepath: str, chunk_size: int = 1024 * 1024) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def main():
    print("=" * 60)
    print("AMAZON ML CHALLENGE 2026 - PHASE 5 FINAL SUBMISSION GENERATION")
    print("=" * 60)
    start_time = time.time()

    os.makedirs(PHASE5_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Preflight checks
    print("\n[Step 1/5] Verifying inputs and checksums...")
    for p, name in [
        (S2_PRED_PATH, "S2 predictions"),
        (S3_PRED_PATH, "S3 predictions"),
        (TEST_S1_PARQUET, "Test S1 parquet"),
        (TEST_S2_PARQUET, "Test S2 parquet"),
        (TEST_S3_PARQUET, "Test S3 parquet"),
    ]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required input file: {name} at {p}")
        print(f"  Found: {name} ({os.path.getsize(p):,} bytes)")

    s2_sha = compute_sha256(S2_PRED_PATH)
    s3_sha = compute_sha256(S3_PRED_PATH)
    print(f"  S2 SHA256: {s2_sha}")
    if s2_sha != EXPECTED_S2_SHA256:
        raise ValueError(f"S2 SHA256 mismatch! Expected {EXPECTED_S2_SHA256}, got {s2_sha}")

    print(f"  S3 SHA256: {s3_sha}")
    if s3_sha != EXPECTED_S3_SHA256:
        raise ValueError(f"S3 SHA256 mismatch! Expected {EXPECTED_S3_SHA256}, got {s3_sha}")

    print("  Input checksum verification PASSED.")

    # 2. Database Connection
    print("\n[Step 2/5] Initializing DuckDB engine...")
    con = duckdb.connect()
    con.execute("SET threads=4")
    con.execute("SET memory_limit='6GB'")
    con.execute("SET preserve_insertion_order=false")

    # Verify S1 count
    test_s1_count = con.execute(f"SELECT count(*) FROM '{TEST_S1_PARQUET}'").fetchone()[0]
    print(f"  Canonical Test S1 entity count: {test_s1_count:,}")
    if test_s1_count != EXPECTED_TEST_S1_COUNT:
        raise ValueError(f"Test S1 count mismatch! Expected {EXPECTED_TEST_S1_COUNT}, got {test_s1_count}")

    # 3. Aggregating positive matches and serializing TSV
    print("\n[Step 3/5] Aggregating positive matches (threshold >= 0.60)...")
    t_gen = time.time()

    # Query logic:
    # 1. Filter positive predictions (predicted_match_label = 1) from S2 and S3
    # 2. Group by source1_entity_id, order candidates deterministically by model_score DESC, candidate_entity_id ASC
    # 3. Aggregate candidate IDs as comma-separated string
    # 4. Left-join from canonical test_s1_entities.parquet to ensure all 1,732,544 S1 entities are included
    # 5. Non-matched entities coalesce to empty string ''
    # 6. Export directly using DuckDB COPY with DELIMITER '\t', HEADER TRUE, QUOTE ''
    gen_sql = f"""
    COPY (
        WITH positives AS (
            SELECT source1_entity_id, candidate_entity_id, model_score
            FROM read_csv('{S2_PRED_PATH}', delim='\\t', header=true, types={{'model_score': 'FLOAT', 'predicted_match_label': 'INT'}})
            WHERE predicted_match_label = 1
            UNION ALL
            SELECT source1_entity_id, candidate_entity_id, model_score
            FROM read_csv('{S3_PRED_PATH}', delim='\\t', header=true, types={{'model_score': 'FLOAT', 'predicted_match_label': 'INT'}})
            WHERE predicted_match_label = 1
        ),
        grouped_matches AS (
            SELECT 
                source1_entity_id,
                string_agg(candidate_entity_id, ',' ORDER BY model_score DESC, candidate_entity_id ASC) AS matched_entity_ids
            FROM positives
            GROUP BY source1_entity_id
        ),
        all_s1 AS (
            SELECT entity_id AS source1_entity_id
            FROM read_parquet('{TEST_S1_PARQUET}')
        )
        SELECT 
            s1.source1_entity_id,
            COALESCE(m.matched_entity_ids, '') AS matched_entity_ids
        FROM all_s1 s1
        LEFT JOIN grouped_matches m ON s1.source1_entity_id = m.source1_entity_id
        ORDER BY s1.source1_entity_id ASC
    ) TO '{SUBMISSION_P5_PATH}' (DELIMITER '\\t', HEADER TRUE, QUOTE '')
    """
    con.execute(gen_sql)
    print(f"  Generated Phase 5 submission at {SUBMISSION_P5_PATH} in {time.time() - t_gen:.2f}s")

    # Also copy to output/matching_results.tsv
    print("\n[Step 4/5] Mirroring submission to output/matching_results.tsv...")
    import shutil
    shutil.copyfile(SUBMISSION_P5_PATH, SUBMISSION_ROOT_PATH)
    print(f"  Mirrored to {SUBMISSION_ROOT_PATH}")

    # 4. Exhaustive Validation
    print("\n[Step 5/5] Performing exhaustive submission validation...")
    val_start = time.time()

    # Check file size & SHA256
    p5_size = os.path.getsize(SUBMISSION_P5_PATH)
    root_size = os.path.getsize(SUBMISSION_ROOT_PATH)
    p5_sha256 = compute_sha256(SUBMISSION_P5_PATH)
    root_sha256 = compute_sha256(SUBMISSION_ROOT_PATH)

    print(f"  Size: {p5_size:,} bytes")
    print(f"  SHA256: {p5_sha256}")
    if p5_sha256 != root_sha256:
        raise ValueError("Checksum mismatch between P2/predictions/phase5 and output/ submissions!")

    # Validate line count and schema via raw file reading
    line_count = 0
    non_empty_count = 0
    empty_count = 0
    bad_lines = 0

    with open(SUBMISSION_P5_PATH, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\r\n")
        if header != "source1_entity_id\tmatched_entity_ids":
            raise ValueError(f"Invalid header! Observed: {repr(header)}, Expected: 'source1_entity_id\\tmatched_entity_ids'")

        for idx, line in enumerate(f, start=1):
            line_count += 1
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) != 2:
                bad_lines += 1
                if bad_lines <= 5:
                    print(f"  [ERROR] Line {idx} does not have exactly 2 tab-separated fields: {repr(line)}")
                continue

            s1_id, matches = parts
            if not s1_id.startswith("S1-"):
                raise ValueError(f"Line {idx} invalid source1_entity_id: {s1_id}")

            if matches == "":
                empty_count += 1
            else:
                non_empty_count += 1
                # Check for accidental quotes or brackets
                if '"' in matches or "'" in matches or "[" in matches or "]" in matches:
                    raise ValueError(f"Line {idx} contains forbidden quote/bracket tokens: {matches[:50]}")

    if bad_lines > 0:
        raise ValueError(f"Found {bad_lines} malformed lines in submission TSV!")

    if line_count != EXPECTED_TEST_S1_COUNT:
        raise ValueError(f"Line count mismatch! Expected {EXPECTED_TEST_S1_COUNT} data rows, found {line_count}")

    print(f"  Total data rows: {line_count:,} (Expected: {EXPECTED_TEST_S1_COUNT:,}) [OK]")
    print(f"  Matched S1 entities: {non_empty_count:,} ({non_empty_count/line_count*100:.2f}%)")
    print(f"  No-match S1 entities: {empty_count:,} ({empty_count/line_count*100:.2f}%)")

    # Target entity ID domain validation using DuckDB
    print("\n  Validating matched target entity IDs against canonical test entity sets...")
    cand_val_query = f"""
    WITH unnested AS (
        SELECT 
            source1_entity_id,
            UNNEST(STRING_SPLIT(matched_entity_ids, ',')) AS target_entity_id
        FROM read_csv('{SUBMISSION_P5_PATH}', delim='\\t', header=true)
        WHERE matched_entity_ids <> ''
    ),
    s2_valid AS (
        SELECT entity_id FROM read_parquet('{TEST_S2_PARQUET}')
    ),
    s3_valid AS (
        SELECT entity_id FROM read_parquet('{TEST_S3_PARQUET}')
    )
    SELECT
        COUNT(*) AS total_matched_pairs,
        COUNT(DISTINCT target_entity_id) AS distinct_targets,
        SUM(CASE WHEN target_entity_id LIKE 'S2-%' THEN 1 ELSE 0 END) AS s2_matches,
        SUM(CASE WHEN target_entity_id LIKE 'S3-%' THEN 1 ELSE 0 END) AS s3_matches,
        SUM(CASE WHEN target_entity_id NOT LIKE 'S2-%' AND target_entity_id NOT LIKE 'S3-%' THEN 1 ELSE 0 END) AS unknown_target_format,
        SUM(CASE WHEN target_entity_id LIKE 'S2-%' AND s2.entity_id IS NULL THEN 1 ELSE 0 END) AS invalid_s2_targets,
        SUM(CASE WHEN target_entity_id LIKE 'S3-%' AND s3.entity_id IS NULL THEN 1 ELSE 0 END) AS invalid_s3_targets
    FROM unnested u
    LEFT JOIN s2_valid s2 ON u.target_entity_id = s2.entity_id
    LEFT JOIN s3_valid s3 ON u.target_entity_id = s3.entity_id
    """
    cand_val = con.execute(cand_val_query).fetchone()
    (
        total_matched_pairs,
        distinct_targets,
        s2_matches,
        s3_matches,
        unknown_target_format,
        invalid_s2_targets,
        invalid_s3_targets,
    ) = cand_val

    print(f"  Total matched pairs in submission: {total_matched_pairs:,}")
    print(f"  Distinct targets: {distinct_targets:,}")
    print(f"  S2 matches: {s2_matches:,}")
    print(f"  S3 matches: {s3_matches:,}")
    print(f"  Unknown target format: {unknown_target_format}")
    print(f"  Invalid S2 targets (not in test S2): {invalid_s2_targets}")
    print(f"  Invalid S3 targets (not in test S3): {invalid_s3_targets}")

    if unknown_target_format > 0 or invalid_s2_targets > 0 or invalid_s3_targets > 0:
        raise ValueError("Matched target entity IDs contained invalid or unknown IDs!")

    print(f"  Exhaustive validation finished in {time.time() - val_start:.2f}s [ALL PASS]")
    print(f"\nFinal Submission SHA256: {p5_sha256}")
    print(f"Total pipeline elapsed time: {time.time() - start_time:.2f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
