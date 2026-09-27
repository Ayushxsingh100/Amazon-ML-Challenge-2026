#!/usr/bin/env python3
"""
Amazon ML Challenge 2026 - Phase 7 Calibrated Submission Generator
===================================================================

Generates the Phase 7 verified competition submission incorporating:
- V4 High-Recall Candidate Retrieval (Strategies A & B)
- 5-Fold Ensemble Model Scoring across all 97M candidates
- Optimal Evaluated Decision Policy: Threshold T=0.88, Score Margin=0.05, Max Target Cap K=8

Outputs:
- output/matching_results.tsv
- P2/predictions/phase7/matching_results.tsv
"""

import os
import sys
import time
import hashlib
import shutil
import duckdb

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Inputs
S2_PRED_PATH = os.path.join(REPO_ROOT, "P2", "predictions", "phase4", "test_predictions_s2.tsv")
S3_PRED_PATH = os.path.join(REPO_ROOT, "P2", "predictions", "phase4", "test_predictions_s3.tsv")
NEW_PRED_PARQUET = os.path.join(REPO_ROOT, "P2", "predictions", "phase7", "new_test_predictions_v4.parquet")

TEST_S1_PARQUET = os.path.join(REPO_ROOT, "P1", "data", "entities", "test", "source1", "test_s1_entities.parquet")
TEST_S2_PARQUET = os.path.join(REPO_ROOT, "P1", "data", "entities", "test", "source2", "test_s2_entities.parquet")
TEST_S3_PARQUET = os.path.join(REPO_ROOT, "P1", "data", "entities", "test", "source3", "test_s3_entities.parquet")

# Outputs
PHASE7_DIR = os.path.join(REPO_ROOT, "P2", "predictions", "phase7")
OUTPUT_DIR = os.path.join(REPO_ROOT, "output")
SUBMISSION_P7_PATH = os.path.join(PHASE7_DIR, "matching_results.tsv")
SUBMISSION_ROOT_PATH = os.path.join(OUTPUT_DIR, "matching_results.tsv")

# Validated Policy Constants
THRESHOLD = 0.88
MARGIN = 0.05
MAX_K = 8
EXPECTED_TEST_S1_COUNT = 1732544


def compute_sha256(filepath: str, chunk_size: int = 1024 * 1024) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def main():
    print("=" * 80)
    print("AMAZON ML CHALLENGE 2026 - PHASE 7 SUBMISSION GENERATOR")
    print(f"Policy: Threshold={THRESHOLD}, Score Margin={MARGIN}, Max Target Cap={MAX_K}")
    print("=" * 80)
    start_time = time.time()

    os.makedirs(PHASE7_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Preflight checks
    print("\n[Step 1/5] Verifying input artifacts...")
    for p, name in [
        (S2_PRED_PATH, "V3 S2 predictions"),
        (S3_PRED_PATH, "V3 S3 predictions"),
        (NEW_PRED_PARQUET, "Incremental V4 test predictions"),
        (TEST_S1_PARQUET, "Test S1 parquet"),
        (TEST_S2_PARQUET, "Test S2 parquet"),
        (TEST_S3_PARQUET, "Test S3 parquet"),
    ]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required input file: {name} at {p}")
        print(f"  Found: {name} ({os.path.getsize(p):,} bytes)")

    # 2. Database Connection
    print("\n[Step 2/5] Initializing DuckDB engine...")
    con = duckdb.connect()
    con.execute("SET threads=4")
    con.execute("SET memory_limit='8GB'")
    con.execute("SET preserve_insertion_order=false")

    test_s1_count = con.execute(f"SELECT count(*) FROM '{TEST_S1_PARQUET}'").fetchone()[0]
    print(f"  Canonical Test S1 entity count: {test_s1_count:,}")
    if test_s1_count != EXPECTED_TEST_S1_COUNT:
        raise ValueError(f"Test S1 count mismatch! Expected {EXPECTED_TEST_S1_COUNT}, got {test_s1_count}")

    # 3. Aggregating candidates and applying Phase 7 policy
    print(f"\n[Step 3/5] Applying Phase 7 policy (T={THRESHOLD}, margin={MARGIN}, max_k={MAX_K})...")
    t_gen = time.time()

    gen_sql = f"""
    COPY (
        WITH cands AS (
            SELECT source1_entity_id, candidate_entity_id, model_score
            FROM read_csv('{S2_PRED_PATH}', delim='\\t', header=true, types={{'model_score': 'FLOAT', 'predicted_match_label': 'INT'}})
            WHERE model_score >= {THRESHOLD}
            UNION ALL
            SELECT source1_entity_id, candidate_entity_id, model_score
            FROM read_csv('{S3_PRED_PATH}', delim='\\t', header=true, types={{'model_score': 'FLOAT', 'predicted_match_label': 'INT'}})
            WHERE model_score >= {THRESHOLD}
            UNION ALL
            SELECT source1_entity_id, candidate_entity_id, model_score
            FROM read_parquet('{NEW_PRED_PARQUET}')
            WHERE model_score >= {THRESHOLD}
        ),
        ranked AS (
            SELECT 
                source1_entity_id, 
                candidate_entity_id, 
                model_score,
                MAX(model_score) OVER (PARTITION BY source1_entity_id) AS top1_score,
                ROW_NUMBER() OVER (PARTITION BY source1_entity_id ORDER BY model_score DESC, candidate_entity_id ASC) AS rank
            FROM cands
        ),
        filtered AS (
            SELECT 
                source1_entity_id,
                candidate_entity_id,
                model_score
            FROM ranked
            WHERE (top1_score - model_score) <= {MARGIN}
              AND rank <= {MAX_K}
        ),
        grouped_matches AS (
            SELECT 
                source1_entity_id,
                string_agg(candidate_entity_id, ',' ORDER BY model_score DESC, candidate_entity_id ASC) AS matched_entity_ids
            FROM filtered
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
    ) TO '{SUBMISSION_P7_PATH}' (DELIMITER '\\t', HEADER TRUE, QUOTE '')
    """

    con.execute(gen_sql)
    print(f"  Generated Phase 7 submission at {SUBMISSION_P7_PATH} in {time.time() - t_gen:.2f}s")

    # Mirror to output/matching_results.tsv
    print("\n[Step 4/5] Mirroring submission to output/matching_results.tsv...")
    shutil.copyfile(SUBMISSION_P7_PATH, SUBMISSION_ROOT_PATH)
    print(f"  Mirrored to {SUBMISSION_ROOT_PATH}")

    # 4. Exhaustive Validation
    print("\n[Step 5/5] Performing exhaustive submission validation...")
    val_start = time.time()

    p7_size = os.path.getsize(SUBMISSION_P7_PATH)
    root_size = os.path.getsize(SUBMISSION_ROOT_PATH)
    p7_sha256 = compute_sha256(SUBMISSION_P7_PATH)
    root_sha256 = compute_sha256(SUBMISSION_ROOT_PATH)

    print(f"  Size: {p7_size:,} bytes")
    print(f"  SHA256: {p7_sha256}")
    if p7_sha256 != root_sha256:
        raise ValueError("Checksum mismatch between P2/predictions/phase7 and output/ submissions!")

    # Check rows, formatting, singletons
    line_count = 0
    non_empty_count = 0
    empty_count = 0
    bad_lines = 0

    with open(SUBMISSION_P7_PATH, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\r\n")
        if header != "source1_entity_id\tmatched_entity_ids":
            raise ValueError(f"Invalid header! Observed: {repr(header)}")

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
                if '"' in matches or "'" in matches or "[" in matches or "]" in matches:
                    raise ValueError(f"Line {idx} contains forbidden quote/bracket tokens: {matches[:50]}")

    if bad_lines > 0:
        raise ValueError(f"Found {bad_lines} malformed lines in submission TSV!")

    if line_count != EXPECTED_TEST_S1_COUNT:
        raise ValueError(f"Line count mismatch! Expected {EXPECTED_TEST_S1_COUNT} data rows, found {line_count}")

    print(f"  Total data rows: {line_count:,} (Expected: {EXPECTED_TEST_S1_COUNT:,}) [OK]")
    print(f"  Matched S1 entities: {non_empty_count:,} ({non_empty_count/line_count*100:.2f}%)")
    print(f"  No-match S1 entities: {empty_count:,} ({empty_count/line_count*100:.2f}%)")

    # Domain integrity check
    print("\n  Validating matched target entity IDs against canonical test entity sets...")
    cand_val_query = f"""
    WITH unnested AS (
        SELECT 
            source1_entity_id,
            UNNEST(STRING_SPLIT(matched_entity_ids, ',')) AS target_entity_id
        FROM read_csv('{SUBMISSION_P7_PATH}', delim='\\t', header=true)
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
    print(f"  Invalid S2 targets: {invalid_s2_targets}")
    print(f"  Invalid S3 targets: {invalid_s3_targets}")

    if unknown_target_format > 0 or invalid_s2_targets > 0 or invalid_s3_targets > 0:
        raise ValueError("Matched target entity IDs contained invalid or unknown IDs!")

    print(f"  Exhaustive validation finished in {time.time() - val_start:.2f}s [ALL PASS]")
    print(f"\nFinal Phase 7 Submission SHA256: {p7_sha256}")
    print(f"Total pipeline elapsed time: {time.time() - start_time:.2f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()
