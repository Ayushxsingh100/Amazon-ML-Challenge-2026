#!/usr/bin/env python3
"""
Analyze fanout control options for V3 combination to ensure p95 <= 300.
"""
import duckdb
import time

con = duckdb.connect()
con.execute("PRAGMA threads=8;")
con.execute("PRAGMA memory_limit='16GB';")

print("Loading GT...")
con.execute("""
    CREATE TABLE gt AS
    SELECT
        source1_entity_id,
        trim(unnest(string_split(matched_entity_ids, ','))) as target_entity_id,
        CASE WHEN target_entity_id LIKE 'S2-%' THEN 's2' ELSE 's3' END as target_source
    FROM read_csv_auto('data/train/train_ground_truth.tsv', delim='\t', header=True);
""")
gt_total = con.execute("SELECT count(*) FROM gt").fetchone()[0]

con.execute("""
    CREATE TABLE s1_entities AS
    SELECT entity_id as source1_entity_id FROM read_parquet('P1/data/entities/train/source1/train_s1_entities.parquet');
""")

for target in ['s2', 's3']:
    v2_path = f"P2/data/candidates/train_candidate_pairs_{target}_v2.tsv"
    exp1_path = f"P1/experiments/phase3/EXP-RET-01/train_candidates_{target}_exp_ret_01.tsv"
    exp2_path = f"P1/experiments/phase3/EXP-RET-02/train_candidates_{target}_exp_ret_02_th080.tsv"
    exp3_path = f"P1/experiments/phase3/EXP-RET-03/train_candidates_{target}_exp_ret_03.tsv"
    exp4_path = f"P1/experiments/phase3/EXP-RET-04/train_candidates_{target}_exp_ret_04.tsv"

    con.execute(f"CREATE TABLE v2_{target} AS SELECT source1_entity_id, matched_entity_id FROM read_csv_auto('{v2_path}', delim='\t', header=True);")
    con.execute(f"CREATE TABLE exp1_{target} AS SELECT source1_entity_id, matched_entity_id FROM read_csv_auto('{exp1_path}', delim='\t', header=True);")
    con.execute(f"CREATE TABLE exp2_{target} AS SELECT source1_entity_id, matched_entity_id FROM read_csv_auto('{exp2_path}', delim='\t', header=True);")
    con.execute(f"CREATE TABLE exp3_{target} AS SELECT source1_entity_id, matched_entity_id FROM read_csv_auto('{exp3_path}', delim='\t', header=True);")
    con.execute(f"CREATE TABLE exp4_{target} AS SELECT source1_entity_id, matched_entity_id FROM read_csv_auto('{exp4_path}', delim='\t', header=True);")

print("Option 1: V2 + EXP-02 + EXP-03 + EXP-04 (Without EXP-01)")
con.execute("""
    CREATE TABLE opt1 AS
    SELECT DISTINCT source1_entity_id, matched_entity_id FROM (
        SELECT source1_entity_id, matched_entity_id FROM v2_s2 UNION ALL
        SELECT source1_entity_id, matched_entity_id FROM exp2_s2 UNION ALL
        SELECT source1_entity_id, matched_entity_id FROM exp3_s2 UNION ALL
        SELECT source1_entity_id, matched_entity_id FROM exp4_s2 UNION ALL
        SELECT source1_entity_id, matched_entity_id FROM v2_s3 UNION ALL
        SELECT source1_entity_id, matched_entity_id FROM exp2_s3 UNION ALL
        SELECT source1_entity_id, matched_entity_id FROM exp3_s3 UNION ALL
        SELECT source1_entity_id, matched_entity_id FROM exp4_s3
    );
""")
c1 = con.execute("SELECT count(*) FROM opt1").fetchone()[0]
cap1 = con.execute("SELECT count(*) FROM opt1 c JOIN gt ON c.source1_entity_id = gt.source1_entity_id AND c.matched_entity_id = gt.target_entity_id").fetchone()[0]
fanout1 = con.execute("""
    SELECT quantile_cont(cand_count, 0.50), quantile_cont(cand_count, 0.90), quantile_cont(cand_count, 0.95), quantile_cont(cand_count, 0.99), max(cand_count), count(CASE WHEN cand_count = 0 THEN 1 END)
    FROM (
        SELECT s1.source1_entity_id, COALESCE(c.cnt, 0) as cand_count
        FROM s1_entities s1
        LEFT JOIN (SELECT source1_entity_id, count(*) as cnt FROM opt1 GROUP BY source1_entity_id) c ON s1.source1_entity_id = c.source1_entity_id
    );
""").fetchone()

print(f"Option 1 Results: Cands={c1:,}, Recall={cap1/gt_total:.4%}, p95={fanout1[2]}, p99={fanout1[3]}, max={fanout1[4]}, zero_S1={fanout1[5]:,}")

print("\nOption 2: EXP-01 with core_name block frequency <= 100")
# Let's inspect frequency distribution of core_name in exp1
con.execute("""
    CREATE TABLE exp1_freq_s2 AS
    SELECT source1_entity_id, matched_entity_id, count(*) OVER (PARTITION BY source1_entity_id) as s1_fan
    FROM exp1_s2;
""")
con.execute("""
    CREATE TABLE exp1_freq_s3 AS
    SELECT source1_entity_id, matched_entity_id, count(*) OVER (PARTITION BY source1_entity_id) as s1_fan
    FROM exp1_s3;
""")

for max_fan in [50, 100, 150]:
    con.execute(f"""
        CREATE OR REPLACE TABLE opt2_{max_fan} AS
        SELECT DISTINCT source1_entity_id, matched_entity_id FROM (
            SELECT source1_entity_id, matched_entity_id FROM v2_s2 UNION ALL
            SELECT source1_entity_id, matched_entity_id FROM exp1_freq_s2 WHERE s1_fan <= {max_fan} UNION ALL
            SELECT source1_entity_id, matched_entity_id FROM exp2_s2 UNION ALL
            SELECT source1_entity_id, matched_entity_id FROM exp3_s2 UNION ALL
            SELECT source1_entity_id, matched_entity_id FROM exp4_s2 UNION ALL
            SELECT source1_entity_id, matched_entity_id FROM v2_s3 UNION ALL
            SELECT source1_entity_id, matched_entity_id FROM exp1_freq_s3 WHERE s1_fan <= {max_fan} UNION ALL
            SELECT source1_entity_id, matched_entity_id FROM exp2_s3 UNION ALL
            SELECT source1_entity_id, matched_entity_id FROM exp3_s3 UNION ALL
            SELECT source1_entity_id, matched_entity_id FROM exp4_s3
        );
    """)
    c2 = con.execute(f"SELECT count(*) FROM opt2_{max_fan}").fetchone()[0]
    cap2 = con.execute(f"SELECT count(*) FROM opt2_{max_fan} c JOIN gt ON c.source1_entity_id = gt.source1_entity_id AND c.matched_entity_id = gt.target_entity_id").fetchone()[0]
    fanout2 = con.execute(f"""
        SELECT quantile_cont(cand_count, 0.50), quantile_cont(cand_count, 0.90), quantile_cont(cand_count, 0.95), quantile_cont(cand_count, 0.99), max(cand_count), count(CASE WHEN cand_count = 0 THEN 1 END)
        FROM (
            SELECT s1.source1_entity_id, COALESCE(c.cnt, 0) as cand_count
            FROM s1_entities s1
            LEFT JOIN (SELECT source1_entity_id, count(*) as cnt FROM opt2_{max_fan} GROUP BY source1_entity_id) c ON s1.source1_entity_id = c.source1_entity_id
        );
    """).fetchone()
    print(f"Option 2 (max_exp1_fan={max_fan}) Results: Cands={c2:,}, Recall={cap2/gt_total:.4%}, p95={fanout2[2]}, p99={fanout2[3]}, max={fanout2[4]}, zero_S1={fanout2[5]:,}")
