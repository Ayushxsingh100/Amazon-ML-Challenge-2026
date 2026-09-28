"""
P2 Phase 2 - Training Candidate Recall (Strategy B)
=====================================================
Generates Strategy B training candidates in DuckDB and computes recall
against ground truth. Writes recall report JSON + distribution TSV.

Strategy B:
  Rule A: country + prefix4 + house
  Rule B: country + exact business_address
  Final = UNION (deduped)
"""
import json
import os
import time
import duckdb

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
NORM_DIR = os.path.join(REPO, "outputs", "person1_step1", "normalized")
GT_PATH = os.path.join(REPO, "outputs", "person1_step1", "train_ground_truth_reconstructed.tsv")
S1_PATH = os.path.join(NORM_DIR, "train_source1_normalized.tsv")
S2_PATH = os.path.join(NORM_DIR, "train_source2_normalized.tsv")
S3_PATH = os.path.join(NORM_DIR, "train_source3_normalized.tsv")

REPORT_DIR = os.path.join(REPO, "P2", "reports")
RECALL_JSON = os.path.join(REPORT_DIR, "strategy_b_train_recall.json")
RECALL_TSV = os.path.join(REPORT_DIR, "strategy_b_train_recall.tsv")
CAND_DIR = os.path.join(REPO, "P2", "data", "candidates")
DB_PATH = os.path.join(REPO, "P2", "data", "train_recall.duckdb")
TMP_DIR = os.path.join(REPO, "P2", "data", "duckdb_train_tmp")

os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(CAND_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)


def main():
    t_start = time.time()

    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    con = duckdb.connect(DB_PATH)
    con.execute("SET memory_limit='4GB'")
    con.execute("SET threads=4")
    con.execute("SET preserve_insertion_order=false")
    con.execute(f"SET temp_directory='{TMP_DIR}'")

    # ----------------------------------------------------------------
    # Load normalized sources with blocking keys
    # ----------------------------------------------------------------
    print("Loading sources...")
    t0 = time.time()
    for name, path in [("s1", S1_PATH), ("s2", S2_PATH), ("s3", S3_PATH)]:
        con.execute(f"""
            CREATE TABLE {name} AS
            SELECT
                entity_id,
                business_name,
                business_address,
                country,
                LEFT(TRIM(business_name), 4) AS prefix4,
                regexp_extract(
                    TRIM(business_address),
                    '\\b[0-9]+[A-Za-z]?\\b',
                    0
                ) AS house
            FROM read_csv(
                '{path}',
                delim='\\t',
                header=true,
                all_varchar=true
            )
        """)
        cnt = con.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
        print(f"  {name.upper()}: {cnt:,} rows")

    print(f"Sources loaded ({time.time() - t0:.1f}s)")

    # ----------------------------------------------------------------
    # Load & explode ground truth
    # ----------------------------------------------------------------
    print("\nLoading ground truth...")
    t0 = time.time()
    con.execute(f"""
        CREATE TABLE gt_exploded AS
        SELECT
            source1_entity_id,
            TRIM(UNNEST(string_split(matched_entity_ids, ','))) AS matched_entity_id
        FROM read_csv(
            '{GT_PATH}',
            delim='\\t',
            header=true,
            all_varchar=true
        )
        WHERE matched_entity_ids IS NOT NULL AND matched_entity_ids <> ''
    """)
    for tgt in ["s2", "s3"]:
        prefix = tgt.upper()
        con.execute(f"""
            CREATE TABLE gt_{tgt} AS
            SELECT DISTINCT source1_entity_id, matched_entity_id
            FROM gt_exploded
            WHERE matched_entity_id LIKE '{prefix}-%'
        """)
        cnt = con.execute(f"SELECT COUNT(*) FROM gt_{tgt}").fetchone()[0]
        print(f"  GT {tgt.upper()} pairs: {cnt:,}")
    print(f"  ({time.time() - t0:.1f}s)")

    # ----------------------------------------------------------------
    # Generate Strategy B candidates & evaluate recall
    # ----------------------------------------------------------------
    results = {}
    for tgt in ["s2", "s3"]:
        t_upper = tgt.upper()
        gt_tbl = f"gt_{tgt}"

        print(f"\n{'='*60}")
        print(f"Strategy B candidates for S1 -> {t_upper}")
        print(f"{'='*60}")
        t0 = time.time()

        # Rule A: country + prefix4 + house
        # Rule B: country + exact address
        con.execute(f"""
            CREATE OR REPLACE TABLE cand_{tgt} AS
            SELECT DISTINCT
                s1.entity_id AS source1_entity_id,
                {tgt}.entity_id AS matched_entity_id
            FROM s1
            JOIN {tgt}
              ON s1.country <> ''
             AND s1.country = {tgt}.country
             AND s1.prefix4 <> ''
             AND s1.prefix4 = {tgt}.prefix4
             AND s1.house <> ''
             AND s1.house = {tgt}.house

            UNION

            SELECT DISTINCT
                s1.entity_id,
                {tgt}.entity_id
            FROM s1
            JOIN {tgt}
              ON s1.country <> ''
             AND s1.country = {tgt}.country
             AND s1.business_address <> ''
             AND s1.business_address = {tgt}.business_address
        """)

        total_cand = con.execute(f"SELECT COUNT(*) FROM cand_{tgt}").fetchone()[0]
        print(f"  Candidates generated: {total_cand:,} ({time.time() - t0:.1f}s)")

        # Ground truth stats
        gt_total = con.execute(f"SELECT COUNT(*) FROM {gt_tbl}").fetchone()[0]

        # True pairs captured
        tp = con.execute(f"""
            SELECT COUNT(*) FROM cand_{tgt} c
            JOIN {gt_tbl} g
              ON c.source1_entity_id = g.source1_entity_id
             AND c.matched_entity_id = g.matched_entity_id
        """).fetchone()[0]

        recall = tp / gt_total if gt_total > 0 else 0.0
        precision = tp / total_cand if total_cand > 0 else 0.0

        # Unique S1 coverage
        s1_total = con.execute("SELECT COUNT(*) FROM s1").fetchone()[0]
        s1_in_cand = con.execute(f"SELECT COUNT(DISTINCT source1_entity_id) FROM cand_{tgt}").fetchone()[0]
        s1_zero = s1_total - s1_in_cand

        # Fully-covered S1 (all GT targets captured)
        con.execute(f"""
            CREATE OR REPLACE TABLE s1_recall_{tgt} AS
            SELECT
                g.source1_entity_id,
                COUNT(*) AS gt_count,
                SUM(CASE WHEN c.matched_entity_id IS NOT NULL THEN 1 ELSE 0 END) AS captured_count
            FROM {gt_tbl} g
            LEFT JOIN cand_{tgt} c
              ON g.source1_entity_id = c.source1_entity_id
             AND g.matched_entity_id = c.matched_entity_id
            GROUP BY g.source1_entity_id
        """)
        fully_covered = con.execute(f"""
            SELECT COUNT(*) FROM s1_recall_{tgt}
            WHERE gt_count = captured_count
        """).fetchone()[0]
        gt_s1_count = con.execute(f"SELECT COUNT(*) FROM s1_recall_{tgt}").fetchone()[0]

        # Candidate count distribution
        con.execute(f"""
            CREATE OR REPLACE TABLE cand_dist_{tgt} AS
            SELECT source1_entity_id, COUNT(*) AS candidate_count
            FROM cand_{tgt}
            GROUP BY source1_entity_id
        """)
        dist_stats = con.execute(f"""
            SELECT
                AVG(candidate_count),
                MEDIAN(candidate_count),
                PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY candidate_count),
                PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY candidate_count),
                PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY candidate_count),
                MAX(candidate_count),
                MIN(candidate_count)
            FROM cand_dist_{tgt}
        """).fetchone()

        # Label candidates for later use
        out_path = os.path.join(CAND_DIR, f"train_strategy_b_candidates_{tgt}.tsv")
        con.execute(f"""
            COPY (
                SELECT
                    c.source1_entity_id,
                    c.matched_entity_id,
                    CASE WHEN g.matched_entity_id IS NOT NULL THEN 1 ELSE 0 END AS label
                FROM cand_{tgt} c
                LEFT JOIN {gt_tbl} g
                  ON c.source1_entity_id = g.source1_entity_id
                 AND c.matched_entity_id = g.matched_entity_id
            )
            TO '{out_path}'
            WITH (FORMAT CSV, DELIMITER '\\t', HEADER)
        """)
        file_size = os.path.getsize(out_path) / (1024 * 1024)

        print(f"  GT true pairs: {gt_total:,}")
        print(f"  Captured true pairs: {tp:,}")
        print(f"  Candidate recall: {recall:.6f} ({recall*100:.4f}%)")
        print(f"  Candidate precision: {precision:.6f} ({precision*100:.4f}%)")
        print(f"  Unique S1 in candidates: {s1_in_cand:,}")
        print(f"  S1 with zero candidates: {s1_zero:,}")
        print(f"  S1 with all GT captured (fully covered): {fully_covered:,}/{gt_s1_count:,}")
        print(f"  Candidates per S1: mean={dist_stats[0]:.2f}, median={dist_stats[1]:.0f}, p90={dist_stats[2]:.0f}, p95={dist_stats[3]:.0f}, p99={dist_stats[4]:.0f}, max={dist_stats[5]:,}")
        print(f"  Labeled candidates saved: {out_path} ({file_size:.1f} MB)")

        results[t_upper] = {
            "gt_total_true_pairs": gt_total,
            "total_candidate_pairs": total_cand,
            "captured_true_pairs": tp,
            "candidate_recall": round(recall, 6),
            "candidate_precision": round(precision, 6),
            "unique_s1_in_candidates": s1_in_cand,
            "s1_with_zero_candidates": s1_zero,
            "fully_covered_s1": fully_covered,
            "total_s1_with_gt": gt_s1_count,
            "total_s1": s1_total,
            "candidate_distribution": {
                "mean": round(dist_stats[0], 2),
                "median": float(dist_stats[1]),
                "p90": float(dist_stats[2]),
                "p95": float(dist_stats[3]),
                "p99": float(dist_stats[4]),
                "max": int(dist_stats[5]),
                "min": int(dist_stats[6]),
            },
            "output_file": out_path,
        }

    # Write recall JSON
    with open(RECALL_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nRecall JSON saved: {RECALL_JSON}")

    # Write recall TSV (summary table)
    with open(RECALL_TSV, "w", encoding="utf-8") as f:
        f.write("metric\tS2\tS3\n")
        for metric in [
            "gt_total_true_pairs",
            "total_candidate_pairs",
            "captured_true_pairs",
            "candidate_recall",
            "candidate_precision",
            "unique_s1_in_candidates",
            "s1_with_zero_candidates",
            "fully_covered_s1",
        ]:
            f.write(f"{metric}\t{results['S2'][metric]}\t{results['S3'][metric]}\n")
    print(f"Recall TSV saved: {RECALL_TSV}")

    con.close()
    if os.path.exists(DB_PATH):
        try:
            os.remove(DB_PATH)
        except OSError:
            pass
    print(f"\nPhase 2 complete ({time.time() - t_start:.1f}s)")


if __name__ == "__main__":
    main()
