"""
P2/scripts/generate_train_candidates.py
========================================
Generates labeled training candidate pairs for the Amazon ML Challenge ER task.

Hybrid Strategy (B + C + D):
  Rule A: country + prefix4 + house                (from P1 Strategy B)
  Rule B: country + exact business_address         (from P1 Strategy B)
  Rule C: country + first word of name + house     (NEW — catches prefix4 misses)
  Rule D: country + exact business_name            (NEW — low volume, easy win)
  Final  = UNION of A, B, C, D (deduplicated via UNION)

Labels each candidate pair:
  1 = pair appears in train_ground_truth_reconstructed.tsv
  0 = pair does not appear

Usage:
  # Full run (all S1 rows):
  python P2/scripts/generate_train_candidates.py

  # Smoke test (first 500 S1 rows):
  python P2/scripts/generate_train_candidates.py --smoke-test

  # Custom limit:
  python P2/scripts/generate_train_candidates.py --smoke-test --limit 1000
"""

import argparse
import os
import sys
import time

import duckdb

# ---------------------------------------------------------------------------
# Paths (relative to repository root)
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
NORM_DIR = os.path.join(REPO_ROOT, "outputs", "person1_step1", "normalized")
GT_PATH = os.path.join(REPO_ROOT, "outputs", "person1_step1", "train_ground_truth_reconstructed.tsv")
OUT_DIR = os.path.join(REPO_ROOT, "P2", "data", "candidates")
TMP_DIR = os.path.join(REPO_ROOT, "P2", "data", "duckdb_train_tmp")
DB_PATH = os.path.join(REPO_ROOT, "P2", "data", "train_candidate_gen.duckdb")

S1_PATH = os.path.join(NORM_DIR, "train_source1_normalized.tsv")
S2_PATH = os.path.join(NORM_DIR, "train_source2_normalized.tsv")
S3_PATH = os.path.join(NORM_DIR, "train_source3_normalized.tsv")

S2_OUT = os.path.join(OUT_DIR, "train_candidate_pairs_s2.tsv")
S3_OUT = os.path.join(OUT_DIR, "train_candidate_pairs_s3.tsv")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate labeled training candidate pairs using hybrid blocking (B+C+D)."
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run in smoke-test mode with a limited number of S1 rows.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Number of S1 rows to use in smoke-test mode (default: 500).",
    )
    return parser.parse_args()


def validate_inputs():
    """Check that all required input files exist before proceeding."""
    missing = []
    for path in [S1_PATH, S2_PATH, S3_PATH, GT_PATH]:
        if not os.path.isfile(path):
            missing.append(path)
    if missing:
        print("ERROR: Missing input files:")
        for p in missing:
            print(f"  {p}")
        sys.exit(1)
    print("All input files found.")


def setup_duckdb(smoke_test: bool):
    """Create a disk-backed DuckDB connection with conservative memory settings."""
    os.makedirs(TMP_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    # Remove old DB to start fresh each run
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    con = duckdb.connect(DB_PATH)
    con.execute("SET memory_limit='4GB'")
    con.execute("SET threads=2")
    con.execute("SET preserve_insertion_order=false")
    con.execute(f"SET temp_directory='{TMP_DIR}'")

    print(f"DuckDB {duckdb.__version__} connected (disk-backed at {DB_PATH})")
    return con


def create_source_tables(con, smoke_test: bool, limit: int):
    """
    Load normalized training sources into DuckDB tables with derived
    blocking keys matching P1's logic plus hybrid extensions.

    prefix4 = LEFT(TRIM(business_name), 4)
    token1  = split_part(TRIM(business_name), ' ', 1)   [NEW for Rule C]
    house   = regexp_extract(TRIM(business_address), '\\b[0-9]+[A-Za-z]?\\b', 0)
    """
    s1_limit_clause = f"LIMIT {limit}" if smoke_test else ""
    s1_parallel_opt = ", parallel=false" if smoke_test else ""

    print(f"\nLoading S1 (train)... {'[SMOKE TEST: ' + str(limit) + ' rows (deterministic)]' if smoke_test else '[FULL]'}")
    t0 = time.time()
    # Note: We use parallel=false + LIMIT for smoke test so row selection is 100%
    # deterministic (exact first N rows of the file). Full run uses default parallel read.
    # S2/S3 are always loaded in full so blocking can match against the full target set.
    con.execute(f"""
        CREATE OR REPLACE TABLE s1 AS
        SELECT
            entity_id,
            business_name,
            business_address,
            country,
            LEFT(TRIM(business_name), 4) AS prefix4,
            split_part(TRIM(business_name), ' ', 1) AS token1,
            regexp_extract(
                TRIM(business_address),
                '[0-9]+[A-Za-z]?',
                0
            ) AS house
        FROM (
            SELECT * FROM read_csv(
                '{S1_PATH}',
                delim='\\t',
                header=true,
                all_varchar=true{s1_parallel_opt}
            )
            {s1_limit_clause}
        )
    """)
    s1_count = con.execute("SELECT COUNT(*) FROM s1").fetchone()[0]
    print(f"  S1 loaded: {s1_count:,} rows ({time.time() - t0:.1f}s)")

    print("Loading S2 (train)...")
    t0 = time.time()
    con.execute(f"""
        CREATE OR REPLACE TABLE s2 AS
        SELECT
            entity_id,
            business_name,
            business_address,
            country,
            LEFT(TRIM(business_name), 4) AS prefix4,
            split_part(TRIM(business_name), ' ', 1) AS token1,
            regexp_extract(
                TRIM(business_address),
                '[0-9]+[A-Za-z]?',
                0
            ) AS house
        FROM read_csv(
            '{S2_PATH}',
            delim='\\t',
            header=true,
            all_varchar=true
        )
    """)
    s2_count = con.execute("SELECT COUNT(*) FROM s2").fetchone()[0]
    print(f"  S2 loaded: {s2_count:,} rows ({time.time() - t0:.1f}s)")

    print("Loading S3 (train)...")
    t0 = time.time()
    con.execute(f"""
        CREATE OR REPLACE TABLE s3 AS
        SELECT
            entity_id,
            business_name,
            business_address,
            country,
            LEFT(TRIM(business_name), 4) AS prefix4,
            split_part(TRIM(business_name), ' ', 1) AS token1,
            regexp_extract(
                TRIM(business_address),
                '[0-9]+[A-Za-z]?',
                0
            ) AS house
        FROM read_csv(
            '{S3_PATH}',
            delim='\\t',
            header=true,
            all_varchar=true
        )
    """)
    s3_count = con.execute("SELECT COUNT(*) FROM s3").fetchone()[0]
    print(f"  S3 loaded: {s3_count:,} rows ({time.time() - t0:.1f}s)")

    return s1_count, s2_count, s3_count


def create_ground_truth_table(con):
    """
    Load ground truth and explode comma-separated matched_entity_ids into
    individual rows for efficient JOIN-based labeling.

    Ground truth format:
      source1_entity_id  matched_entity_ids
      S1-12345           S2-111,S2-222,S3-333

    We split the CSV list using DuckDB's string_split + UNNEST, then filter
    by S2/S3 prefix for each target source.
    """
    print("\nLoading ground truth...")
    t0 = time.time()

    # Load raw ground truth
    con.execute(f"""
        CREATE OR REPLACE TABLE gt_raw AS
        SELECT
            source1_entity_id,
            matched_entity_ids
        FROM read_csv(
            '{GT_PATH}',
            delim='\\t',
            header=true,
            all_varchar=true
        )
    """)
    gt_raw_count = con.execute("SELECT COUNT(*) FROM gt_raw").fetchone()[0]
    print(f"  Ground truth raw rows: {gt_raw_count:,}")

    # Explode comma-separated IDs into individual (s1_id, matched_id) rows
    con.execute("""
        CREATE OR REPLACE TABLE gt_exploded AS
        SELECT
            source1_entity_id,
            TRIM(UNNEST(string_split(matched_entity_ids, ','))) AS matched_entity_id
        FROM gt_raw
        WHERE matched_entity_ids IS NOT NULL
          AND matched_entity_ids <> ''
    """)
    gt_exploded_count = con.execute("SELECT COUNT(*) FROM gt_exploded").fetchone()[0]
    print(f"  Ground truth exploded pairs: {gt_exploded_count:,}")

    # Create S2-specific and S3-specific ground truth for faster JOINs
    con.execute("""
        CREATE OR REPLACE TABLE gt_s2 AS
        SELECT DISTINCT source1_entity_id, matched_entity_id
        FROM gt_exploded
        WHERE matched_entity_id LIKE 'S2-%'
    """)
    gt_s2_count = con.execute("SELECT COUNT(*) FROM gt_s2").fetchone()[0]
    print(f"  GT S2 pairs: {gt_s2_count:,}")

    con.execute("""
        CREATE OR REPLACE TABLE gt_s3 AS
        SELECT DISTINCT source1_entity_id, matched_entity_id
        FROM gt_exploded
        WHERE matched_entity_id LIKE 'S3-%'
    """)
    gt_s3_count = con.execute("SELECT COUNT(*) FROM gt_s3").fetchone()[0]
    print(f"  GT S3 pairs: {gt_s3_count:,}")

    print(f"  ({time.time() - t0:.1f}s)")
    return gt_s2_count, gt_s3_count


def generate_candidates_and_label(con, target: str, out_path: str):
    """
    Generate candidate pairs for S1 -> target (s2 or s3) using the hybrid
    strategy (B + C + D), then label using ground truth.

    Rule A: country + prefix4 + house                (P1 Strategy B)
    Rule B: country + exact business_address         (P1 Strategy B)
    Rule C: country + token1 + house                 (NEW — hybrid)
    Rule D: country + exact business_name            (NEW — hybrid)
    Final = UNION of A, B, C, D (implicit dedup)

    Then LEFT JOIN with ground truth to assign labels.
    """
    t_name = target  # "s2" or "s3"
    gt_table = f"gt_{t_name}"

    print(f"\n{'='*60}")
    print(f"Generating S1 -> {t_name.upper()} candidates (Hybrid B+C+D)...")
    print(f"{'='*60}")
    t0 = time.time()

    # Step 1: Generate blocking candidates
    # Rule A (P1 Strategy B): country + prefix4 + house
    # Rule B (P1 Strategy B): country + exact address
    # Rule C (NEW):           country + token1 + house
    # Rule D (NEW):           country + exact name
    con.execute(f"""
        CREATE OR REPLACE TABLE candidates_{t_name} AS

        -- Rule A: country + prefix4 + house (from P1 Strategy B)
        SELECT DISTINCT
            s1.entity_id AS source1_entity_id,
            {t_name}.entity_id AS matched_entity_id
        FROM s1
        JOIN {t_name}
          ON s1.country <> ''
         AND s1.country = {t_name}.country
         AND s1.prefix4 <> ''
         AND s1.prefix4 = {t_name}.prefix4
         AND s1.house <> ''
         AND s1.house = {t_name}.house

        UNION

        -- Rule B: country + exact address (from P1 Strategy B)
        SELECT DISTINCT
            s1.entity_id,
            {t_name}.entity_id
        FROM s1
        JOIN {t_name}
          ON s1.country <> ''
         AND s1.country = {t_name}.country
         AND s1.business_address <> ''
         AND s1.business_address = {t_name}.business_address

        UNION

        -- Rule C: country + first word of name + house (NEW)
        SELECT DISTINCT
            s1.entity_id,
            {t_name}.entity_id
        FROM s1
        JOIN {t_name}
          ON s1.country <> ''
         AND s1.country = {t_name}.country
         AND s1.token1 <> ''
         AND s1.token1 = {t_name}.token1
         AND s1.house <> ''
         AND s1.house = {t_name}.house

        UNION

        -- Rule D: country + exact business name (NEW)
        SELECT DISTINCT
            s1.entity_id,
            {t_name}.entity_id
        FROM s1
        JOIN {t_name}
          ON s1.country <> ''
         AND s1.country = {t_name}.country
         AND TRIM(s1.business_name) <> ''
         AND TRIM(s1.business_name) = TRIM({t_name}.business_name)
    """)
    cand_count = con.execute(f"SELECT COUNT(*) FROM candidates_{t_name}").fetchone()[0]
    print(f"  Candidate pairs generated: {cand_count:,} ({time.time() - t0:.1f}s)")

    # Step 2: Label candidates using ground truth via LEFT JOIN
    print(f"  Labeling candidates against ground truth...")
    t1 = time.time()
    con.execute(f"""
        CREATE OR REPLACE TABLE labeled_{t_name} AS
        SELECT
            c.source1_entity_id,
            c.matched_entity_id,
            CASE WHEN gt.matched_entity_id IS NOT NULL THEN 1 ELSE 0 END AS label
        FROM candidates_{t_name} c
        LEFT JOIN {gt_table} gt
          ON c.source1_entity_id = gt.source1_entity_id
         AND c.matched_entity_id = gt.matched_entity_id
    """)
    print(f"  Labeling complete ({time.time() - t1:.1f}s)")

    # Step 3: Compute statistics
    stats = con.execute(f"""
        SELECT
            COUNT(*)                                          AS total,
            SUM(CASE WHEN label = 1 THEN 1 ELSE 0 END)       AS positives,
            SUM(CASE WHEN label = 0 THEN 1 ELSE 0 END)       AS negatives,
            COUNT(DISTINCT source1_entity_id)                  AS unique_s1
        FROM labeled_{t_name}
    """).fetchone()
    total, positives, negatives, unique_s1 = stats
    pos_rate = (positives / total * 100) if total > 0 else 0.0

    print(f"\n  --- {t_name.upper()} Statistics ---")
    print(f"  Total candidate pairs : {total:,}")
    print(f"  Positives (label=1)   : {positives:,}")
    print(f"  Negatives (label=0)   : {negatives:,}")
    print(f"  Positive rate         : {pos_rate:.4f}%")
    print(f"  Unique S1 entities    : {unique_s1:,}")

    # Step 4: Export to TSV
    print(f"\n  Exporting to {out_path}...")
    t2 = time.time()
    con.execute(f"""
        COPY labeled_{t_name}
        TO '{out_path}'
        WITH (
            FORMAT CSV,
            DELIMITER '\\t',
            HEADER
        )
    """)
    file_size = os.path.getsize(out_path) / (1024 * 1024)
    print(f"  Saved: {out_path} ({file_size:.1f} MB, {time.time() - t2:.1f}s)")

    return {
        "target": t_name.upper(),
        "total_pairs": total,
        "positives": positives,
        "negatives": negatives,
        "positive_rate": pos_rate,
        "unique_s1": unique_s1,
    }


def main():
    args = parse_args()
    smoke_test = args.smoke_test
    limit = args.limit

    print("=" * 60)
    print("P2: Training Candidate Generator (Hybrid B+C+D)")
    print(f"Mode: {'SMOKE TEST (' + str(limit) + ' S1 rows)' if smoke_test else 'FULL RUN'}")
    print("=" * 60)

    # Validate inputs
    validate_inputs()

    # Setup DuckDB
    con = setup_duckdb(smoke_test)

    try:
        # Load source tables
        s1_count, s2_count, s3_count = create_source_tables(con, smoke_test, limit)

        # Load and explode ground truth
        gt_s2_count, gt_s3_count = create_ground_truth_table(con)

        # Generate S2 candidates
        stats_s2 = generate_candidates_and_label(con, "s2", S2_OUT)

        # Generate S3 candidates
        stats_s3 = generate_candidates_and_label(con, "s3", S3_OUT)

        # Final summary
        print("\n" + "=" * 60)
        print("FINAL SUMMARY")
        print("=" * 60)
        print(f"  S1 rows used           : {s1_count:,}")
        print(f"  S2 rows                : {s2_count:,}")
        print(f"  S3 rows                : {s3_count:,}")
        print(f"  GT S2 pairs            : {gt_s2_count:,}")
        print(f"  GT S3 pairs            : {gt_s3_count:,}")
        print()
        for stats in [stats_s2, stats_s3]:
            print(f"  [{stats['target']}]")
            print(f"    Total candidate pairs : {stats['total_pairs']:,}")
            print(f"    Positives (label=1)   : {stats['positives']:,}")
            print(f"    Negatives (label=0)   : {stats['negatives']:,}")
            print(f"    Positive rate         : {stats['positive_rate']:.4f}%")
            print(f"    Unique S1 entities    : {stats['unique_s1']:,}")
            print()

    finally:
        con.close()
        # Cleanup temp DB file (data is exported to TSV)
        if os.path.exists(DB_PATH):
            try:
                os.remove(DB_PATH)
            except OSError:
                pass
        print("Done.")


if __name__ == "__main__":
    main()
