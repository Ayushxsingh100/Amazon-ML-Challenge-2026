#!/usr/bin/env python3
"""
Evaluate combination of V2 + Phase 3 retrieval experiments on Train set.
Calculates exact incremental attribution, combined recall, fanout distribution, and zero-candidate S1 count.
"""
import duckdb
import time
import json
from pathlib import Path

def main():
    con = duckdb.connect()
    con.execute("PRAGMA threads=8;")
    con.execute("PRAGMA memory_limit='16GB';")

    print("1. Loading ground truth...")
    con.execute("""
        CREATE TABLE gt AS
        SELECT
            source1_entity_id,
            trim(unnest(string_split(matched_entity_ids, ','))) as target_entity_id,
            CASE WHEN target_entity_id LIKE 'S2-%' THEN 's2' ELSE 's3' END as target_source
        FROM read_csv_auto('data/train/train_ground_truth.tsv', delim='\t', header=True);
    """)
    gt_total = con.execute("SELECT count(*) FROM gt").fetchone()[0]
    gt_s2 = con.execute("SELECT count(*) FROM gt WHERE target_source = 's2'").fetchone()[0]
    gt_s3 = con.execute("SELECT count(*) FROM gt WHERE target_source = 's3'").fetchone()[0]
    print(f"Ground truth loaded: Total={gt_total:,}, S2={gt_s2:,}, S3={gt_s3:,}")

    # S1 entities for zero-candidate tracking
    con.execute("""
        CREATE TABLE s1_entities AS
        SELECT entity_id as source1_entity_id FROM read_parquet('P1/data/entities/train/source1/train_s1_entities.parquet');
    """)
    s1_total = con.execute("SELECT count(*) FROM s1_entities").fetchone()[0]
    print(f"Total S1 entities: {s1_total:,}")

    for target in ['s2', 's3']:
        print(f"\n================ Evaluating {target.upper()} ================")
        v2_path = f"P2/data/candidates/train_candidate_pairs_{target}_v2.tsv"
        exp1_path = f"P1/experiments/phase3/EXP-RET-01/train_candidates_{target}_exp_ret_01.tsv"
        exp2_path = f"P1/experiments/phase3/EXP-RET-02/train_candidates_{target}_exp_ret_02_th080.tsv"
        exp3_path = f"P1/experiments/phase3/EXP-RET-03/train_candidates_{target}_exp_ret_03.tsv"
        exp4_path = f"P1/experiments/phase3/EXP-RET-04/train_candidates_{target}_exp_ret_04.tsv"

        # Load tables
        print(f"Loading candidate sources for {target}...")
        con.execute(f"CREATE TABLE v2_{target} AS SELECT source1_entity_id, matched_entity_id FROM read_csv_auto('{v2_path}', delim='\t', header=True);")
        con.execute(f"CREATE TABLE exp1_{target} AS SELECT source1_entity_id, matched_entity_id FROM read_csv_auto('{exp1_path}', delim='\t', header=True);")
        con.execute(f"CREATE TABLE exp2_{target} AS SELECT source1_entity_id, matched_entity_id FROM read_csv_auto('{exp2_path}', delim='\t', header=True);")
        con.execute(f"CREATE TABLE exp3_{target} AS SELECT source1_entity_id, matched_entity_id FROM read_csv_auto('{exp3_path}', delim='\t', header=True);")
        con.execute(f"CREATE TABLE exp4_{target} AS SELECT source1_entity_id, matched_entity_id FROM read_csv_auto('{exp4_path}', delim='\t', header=True);")

        # Combined table
        print(f"Combining all candidates for {target}...")
        t0 = time.time()
        con.execute(f"""
            CREATE TABLE v3_{target} AS
            SELECT DISTINCT source1_entity_id, matched_entity_id FROM (
                SELECT source1_entity_id, matched_entity_id FROM v2_{target}
                UNION ALL
                SELECT source1_entity_id, matched_entity_id FROM exp1_{target}
                UNION ALL
                SELECT source1_entity_id, matched_entity_id FROM exp2_{target}
                UNION ALL
                SELECT source1_entity_id, matched_entity_id FROM exp3_{target}
                UNION ALL
                SELECT source1_entity_id, matched_entity_id FROM exp4_{target}
            );
        """)
        v3_count = con.execute(f"SELECT count(*) FROM v3_{target}").fetchone()[0]
        print(f"Total V3 unique candidates for {target}: {v3_count:,} (computed in {time.time()-t0:.2f}s)")

        # Incremental attribution
        print("Computing exact incremental attribution...")
        # V2 captured
        v2_cap = con.execute(f"""
            SELECT count(*) FROM v2_{target} c
            JOIN gt ON c.source1_entity_id = gt.source1_entity_id AND c.matched_entity_id = gt.target_entity_id
            WHERE gt.target_source = '{target}';
        """).fetchone()[0]

        # EXP-01 new captures (not in V2)
        exp1_new = con.execute(f"""
            SELECT count(*) FROM (
                SELECT e.source1_entity_id, e.matched_entity_id FROM exp1_{target} e
                ANTI JOIN v2_{target} v ON e.source1_entity_id = v.source1_entity_id AND e.matched_entity_id = v.matched_entity_id
            ) c
            JOIN gt ON c.source1_entity_id = gt.source1_entity_id AND c.matched_entity_id = gt.target_entity_id
            WHERE gt.target_source = '{target}';
        """).fetchone()[0]

        # EXP-02 new captures (not in V2, not in EXP-01)
        exp2_new = con.execute(f"""
            SELECT count(*) FROM (
                SELECT e.source1_entity_id, e.matched_entity_id FROM exp2_{target} e
                ANTI JOIN v2_{target} v ON e.source1_entity_id = v.source1_entity_id AND e.matched_entity_id = v.matched_entity_id
                ANTI JOIN exp1_{target} e1 ON e.source1_entity_id = e1.source1_entity_id AND e.matched_entity_id = e1.matched_entity_id
            ) c
            JOIN gt ON c.source1_entity_id = gt.source1_entity_id AND c.matched_entity_id = gt.target_entity_id
            WHERE gt.target_source = '{target}';
        """).fetchone()[0]

        # EXP-03 new captures (not in V2, not in EXP-01, not in EXP-02)
        exp3_new = con.execute(f"""
            SELECT count(*) FROM (
                SELECT e.source1_entity_id, e.matched_entity_id FROM exp3_{target} e
                ANTI JOIN v2_{target} v ON e.source1_entity_id = v.source1_entity_id AND e.matched_entity_id = v.matched_entity_id
                ANTI JOIN exp1_{target} e1 ON e.source1_entity_id = e1.source1_entity_id AND e.matched_entity_id = e1.matched_entity_id
                ANTI JOIN exp2_{target} e2 ON e.source1_entity_id = e2.source1_entity_id AND e.matched_entity_id = e2.matched_entity_id
            ) c
            JOIN gt ON c.source1_entity_id = gt.source1_entity_id AND c.matched_entity_id = gt.target_entity_id
            WHERE gt.target_source = '{target}';
        """).fetchone()[0]

        # EXP-04 new captures (not in V2, not in EXP-01, not in EXP-02, not in EXP-03)
        exp4_new = con.execute(f"""
            SELECT count(*) FROM (
                SELECT e.source1_entity_id, e.matched_entity_id FROM exp4_{target} e
                ANTI JOIN v2_{target} v ON e.source1_entity_id = v.source1_entity_id AND e.matched_entity_id = v.matched_entity_id
                ANTI JOIN exp1_{target} e1 ON e.source1_entity_id = e1.source1_entity_id AND e.matched_entity_id = e1.matched_entity_id
                ANTI JOIN exp2_{target} e2 ON e.source1_entity_id = e2.source1_entity_id AND e.matched_entity_id = e2.matched_entity_id
                ANTI JOIN exp3_{target} e3 ON e.source1_entity_id = e3.source1_entity_id AND e.matched_entity_id = e3.matched_entity_id
            ) c
            JOIN gt ON c.source1_entity_id = gt.source1_entity_id AND c.matched_entity_id = gt.target_entity_id
            WHERE gt.target_source = '{target}';
        """).fetchone()[0]

        # Total V3 captured
        v3_cap = con.execute(f"""
            SELECT count(*) FROM v3_{target} c
            JOIN gt ON c.source1_entity_id = gt.source1_entity_id AND c.matched_entity_id = gt.target_entity_id
            WHERE gt.target_source = '{target}';
        """).fetchone()[0]

        target_gt = gt_s2 if target == 's2' else gt_s3
        recall = v3_cap / target_gt
        print(f"Incremental Attribution for {target.upper()}:")
        print(f"  V2 captured:        {v2_cap:,} ({v2_cap/target_gt:.4%})")
        print(f"  + EXP-01 new:       {exp1_new:,}")
        print(f"  + EXP-02 new:       {exp2_new:,}")
        print(f"  + EXP-03 new:       {exp3_new:,}")
        print(f"  + EXP-04 new:       {exp4_new:,}")
        print(f"  = V3 total captured:{v3_cap:,} ({recall:.4%})")
        print(f"  Check sum identity: {v2_cap + exp1_new + exp2_new + exp3_new + exp4_new:,} == {v3_cap:,}")
        assert v2_cap + exp1_new + exp2_new + exp3_new + exp4_new == v3_cap, "Sum mismatch in attribution!"

    # Overall Combined V3 evaluation
    print("\n================ COMBINED V3 OVERALL EVALUATION ================")
    con.execute("""
        CREATE TABLE v3_all AS
        SELECT source1_entity_id, matched_entity_id FROM v3_s2
        UNION ALL
        SELECT source1_entity_id, matched_entity_id FROM v3_s3;
    """)
    total_cands = con.execute("SELECT count(*) FROM v3_all").fetchone()[0]
    total_captured = con.execute("""
        SELECT count(*) FROM v3_all c
        JOIN gt ON c.source1_entity_id = gt.source1_entity_id AND c.matched_entity_id = gt.target_entity_id;
    """).fetchone()[0]
    combined_recall = total_captured / gt_total
    missed_pairs = gt_total - total_captured

    # Fanout distribution across all S1 entities
    con.execute("""
        CREATE TABLE fanout AS
        SELECT s1.source1_entity_id, COALESCE(c.cnt, 0) as cand_count
        FROM s1_entities s1
        LEFT JOIN (
            SELECT source1_entity_id, count(*) as cnt
            FROM v3_all
            GROUP BY source1_entity_id
        ) c ON s1.source1_entity_id = c.source1_entity_id;
    """)

    zero_cands = con.execute("SELECT count(*) FROM fanout WHERE cand_count = 0").fetchone()[0]
    p50, p90, p95, p99, p999, max_f = con.execute("""
        SELECT
            quantile_cont(cand_count, 0.50),
            quantile_cont(cand_count, 0.90),
            quantile_cont(cand_count, 0.95),
            quantile_cont(cand_count, 0.99),
            quantile_cont(cand_count, 0.999),
            max(cand_count)
        FROM fanout;
    """).fetchone()

    print(f"\nFinal V3 Combined Metrics:")
    print(f"  Total Candidates:   {total_cands:,} (V2: 67,332,524, Delta: +{total_cands - 67332524:,})")
    print(f"  Captured True Pairs:{total_captured:,} (V2: 5,023,168, Delta: +{total_captured - 5023168:,})")
    print(f"  Missed True Pairs:  {missed_pairs:,} (V2: 2,615,197, Delta: -{2615197 - missed_pairs:,})")
    print(f"  Combined Recall:    {combined_recall:.6%} (V2: 65.7623%, Delta: +{combined_recall - 0.657623:.4%})")
    print(f"  Zero-candidate S1:  {zero_cands:,} (V2: 94,043, Delta: -{94043 - zero_cands:,})")
    print(f"  Fanout p50:         {p50:.1f}")
    print(f"  Fanout p90:         {p90:.1f}")
    print(f"  Fanout p95:         {p95:.1f} (Target <= 300: {'PASS' if p95 <= 300 else 'FAIL'})")
    print(f"  Fanout p99:         {p99:.1f}")
    print(f"  Fanout p99.9:       {p999:.1f}")
    print(f"  Fanout max:         {max_f}")

    res = {
        "total_candidates": total_cands,
        "captured_true_pairs": total_captured,
        "missed_true_pairs": missed_pairs,
        "combined_recall": combined_recall,
        "zero_candidate_s1": zero_cands,
        "fanout": {
            "p50": p50,
            "p90": p90,
            "p95": p95,
            "p99": p99,
            "p999": p999,
            "max": max_f
        }
    }
    with open("P1/reports/v3_combination_metrics.json", "w") as f:
        json.dump(res, f, indent=2)
    print("Saved metrics to P1/reports/v3_combination_metrics.json")

if __name__ == "__main__":
    main()
