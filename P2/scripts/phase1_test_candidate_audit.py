"""
P2 Phase 1 - Test Candidate Audit
==================================
Validates test candidate pair TSVs, computes distributions, writes audit JSON + TSV.
"""
import json
import os
import time
import duckdb

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
S2_CAND = os.path.join(REPO, "outputs", "person1_step1", "test_candidate_pairs_s2.tsv")
S3_CAND = os.path.join(REPO, "outputs", "person1_step1", "test_candidate_pairs_s3.tsv")
REPORT_DIR = os.path.join(REPO, "P2", "reports")
AUDIT_JSON = os.path.join(REPORT_DIR, "strategy_b_test_candidate_audit.json")
DIST_TSV = os.path.join(REPORT_DIR, "strategy_b_test_candidate_distribution.tsv")

# All S1 IDs from test source 1
TEST_S1 = os.path.join(REPO, "outputs", "person1_step1", "normalized", "test_source1_normalized.tsv")

os.makedirs(REPORT_DIR, exist_ok=True)


def audit_candidate_file(con, path, target_prefix, label):
    """Audit a single candidate file and return metrics dict."""
    print(f"\n{'='*60}")
    print(f"Auditing {label}: {os.path.basename(path)}")
    print(f"{'='*60}")

    # Existence
    exists = os.path.isfile(path)
    if not exists:
        return {"file": os.path.basename(path), "exists": False}

    file_size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"  File size: {file_size_mb:.1f} MB")

    # Load into table
    tbl = f"cand_{label.lower()}"
    con.execute(f"""
        CREATE OR REPLACE TABLE {tbl} AS
        SELECT * FROM read_csv(
            '{path}',
            delim='\\t',
            header=true,
            all_varchar=true
        )
    """)

    # Header check
    cols = [desc[0] for desc in con.execute(f"SELECT * FROM {tbl} LIMIT 0").description]
    print(f"  Header: {cols}")
    expected_header = ["source1_entity_id", "matched_entity_id"]
    header_ok = cols == expected_header

    # Row count
    total_rows = con.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
    print(f"  Total rows: {total_rows:,}")

    # Distinct IDs
    distinct_s1 = con.execute(f"SELECT COUNT(DISTINCT source1_entity_id) FROM {tbl}").fetchone()[0]
    distinct_target = con.execute(f"SELECT COUNT(DISTINCT matched_entity_id) FROM {tbl}").fetchone()[0]
    print(f"  Distinct S1 IDs: {distinct_s1:,}")
    print(f"  Distinct {target_prefix} IDs: {distinct_target:,}")

    # Duplicate pairs
    dup_pairs = con.execute(f"""
        SELECT COUNT(*) FROM (
            SELECT source1_entity_id, matched_entity_id, COUNT(*) AS cnt
            FROM {tbl}
            GROUP BY source1_entity_id, matched_entity_id
            HAVING cnt > 1
        )
    """).fetchone()[0]
    print(f"  Duplicate pairs: {dup_pairs:,}")

    # Empty/null IDs
    null_s1 = con.execute(f"""
        SELECT COUNT(*) FROM {tbl}
        WHERE source1_entity_id IS NULL OR TRIM(source1_entity_id) = ''
    """).fetchone()[0]
    null_target = con.execute(f"""
        SELECT COUNT(*) FROM {tbl}
        WHERE matched_entity_id IS NULL OR TRIM(matched_entity_id) = ''
    """).fetchone()[0]
    print(f"  Null/empty S1 IDs: {null_s1:,}")
    print(f"  Null/empty target IDs: {null_target:,}")

    # Prefix checks
    bad_s1_prefix = con.execute(f"""
        SELECT COUNT(*) FROM {tbl}
        WHERE source1_entity_id NOT LIKE 'S1-%'
    """).fetchone()[0]
    bad_target_prefix = con.execute(f"""
        SELECT COUNT(*) FROM {tbl}
        WHERE matched_entity_id NOT LIKE '{target_prefix}-%'
    """).fetchone()[0]
    s1_to_s1 = con.execute(f"""
        SELECT COUNT(*) FROM {tbl}
        WHERE matched_entity_id LIKE 'S1-%'
    """).fetchone()[0]
    print(f"  S1 IDs without S1- prefix: {bad_s1_prefix:,}")
    print(f"  Target IDs without {target_prefix}- prefix: {bad_target_prefix:,}")
    print(f"  S1->S1 self-pairs: {s1_to_s1:,}")

    # Distribution per S1
    con.execute(f"""
        CREATE OR REPLACE TABLE dist_{label.lower()} AS
        SELECT source1_entity_id, COUNT(*) AS candidate_count
        FROM {tbl}
        GROUP BY source1_entity_id
    """)
    stats = con.execute(f"""
        SELECT
            AVG(candidate_count) AS mean_cand,
            MEDIAN(candidate_count) AS median_cand,
            PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY candidate_count) AS p90,
            PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY candidate_count) AS p95,
            PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY candidate_count) AS p99,
            MAX(candidate_count) AS max_cand,
            MIN(candidate_count) AS min_cand
        FROM dist_{label.lower()}
    """).fetchone()

    mean_c, median_c, p90, p95, p99, max_c, min_c = stats
    print(f"  Candidates per S1: mean={mean_c:.2f}, median={median_c:.0f}, p90={p90:.0f}, p95={p95:.0f}, p99={p99:.0f}, max={max_c:,}")

    # S1 with zero candidates (from test_source1)
    total_test_s1 = con.execute(f"""
        SELECT COUNT(DISTINCT entity_id) FROM read_csv(
            '{TEST_S1}',
            delim='\\t',
            header=true,
            all_varchar=true
        )
    """).fetchone()[0]

    zero_cand_s1 = total_test_s1 - distinct_s1
    print(f"  Total test S1 entities: {total_test_s1:,}")
    print(f"  S1 with zero candidates: {zero_cand_s1:,}")

    return {
        "file": os.path.basename(path),
        "target_source": target_prefix,
        "file_exists": exists,
        "file_size_mb": round(file_size_mb, 1),
        "header": cols,
        "header_correct": header_ok,
        "total_rows": total_rows,
        "distinct_s1": distinct_s1,
        "distinct_target": distinct_target,
        "duplicate_pairs": dup_pairs,
        "null_empty_s1_ids": null_s1,
        "null_empty_target_ids": null_target,
        "bad_s1_prefix": bad_s1_prefix,
        "bad_target_prefix": bad_target_prefix,
        "s1_to_s1_pairs": s1_to_s1,
        "total_test_s1_entities": total_test_s1,
        "s1_with_zero_candidates": zero_cand_s1,
        "candidates_per_s1": {
            "mean": round(mean_c, 2),
            "median": float(median_c),
            "p90": float(p90),
            "p95": float(p95),
            "p99": float(p99),
            "max": int(max_c),
            "min": int(min_c),
        },
    }


def main():
    t_start = time.time()
    con = duckdb.connect()
    con.execute("SET memory_limit='4GB'")
    con.execute("SET threads=4")

    results = {}
    for path, prefix, label in [
        (S2_CAND, "S2", "S2"),
        (S3_CAND, "S3", "S3"),
    ]:
        results[label] = audit_candidate_file(con, path, prefix, label)

    # Write audit JSON
    with open(AUDIT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nAudit JSON saved: {AUDIT_JSON}")

    # Write distribution TSV combining both
    con.execute(f"""
        COPY (
            SELECT 'S2' AS target_source, source1_entity_id, candidate_count
            FROM dist_s2
            UNION ALL
            SELECT 'S3' AS target_source, source1_entity_id, candidate_count
            FROM dist_s3
            ORDER BY target_source, candidate_count DESC
        )
        TO '{DIST_TSV}'
        WITH (FORMAT CSV, DELIMITER '\\t', HEADER)
    """)
    print(f"Distribution TSV saved: {DIST_TSV}")

    con.close()
    print(f"\nPhase 1 complete ({time.time() - t_start:.1f}s)")


if __name__ == "__main__":
    main()
