#!/usr/bin/env python3
"""
Compute exact incremental attribution for V3 candidate generation.
"""
import duckdb
import json

con = duckdb.connect()
con.execute('PRAGMA threads=8;')
con.execute("PRAGMA memory_limit='12GB';")

con.execute("""
    CREATE TABLE gt AS
    SELECT
        source1_entity_id,
        trim(unnest(string_split(matched_entity_ids, ','))) as target_entity_id,
        CASE WHEN target_entity_id LIKE 'S2-%' THEN 's2' ELSE 's3' END as target_source
    FROM read_csv_auto('data/train/train_ground_truth.tsv', delim='\t', header=True);
""")

attr = {}
for target in ['s2', 's3']:
    v2_path = f'P2/data/candidates/train_candidate_pairs_{target}_v2.tsv'
    exp1_path = f'P1/experiments/phase3/EXP-RET-01/train_candidates_{target}_exp_ret_01.tsv'
    exp2_path = f'P1/experiments/phase3/EXP-RET-02/train_candidates_{target}_exp_ret_02_th080.tsv'
    exp3_path = f'P1/experiments/phase3/EXP-RET-03/train_candidates_{target}_exp_ret_03.tsv'
    exp4_path = f'P1/experiments/phase3/EXP-RET-04/train_candidates_{target}_exp_ret_04.tsv'

    con.execute(f"CREATE TABLE v2_{target} AS SELECT source1_entity_id, matched_entity_id FROM read_csv_auto('{v2_path}', delim='\t', header=True);")
    con.execute(f"CREATE TABLE exp1_{target} AS SELECT source1_entity_id, matched_entity_id, count(*) OVER (PARTITION BY source1_entity_id) as s1_fan FROM read_csv_auto('{exp1_path}', delim='\t', header=True);")
    con.execute(f"CREATE TABLE exp2_{target} AS SELECT source1_entity_id, matched_entity_id FROM read_csv_auto('{exp2_path}', delim='\t', header=True);")
    con.execute(f"CREATE TABLE exp3_{target} AS SELECT source1_entity_id, matched_entity_id FROM read_csv_auto('{exp3_path}', delim='\t', header=True);")
    con.execute(f"CREATE TABLE exp4_{target} AS SELECT source1_entity_id, matched_entity_id FROM read_csv_auto('{exp4_path}', delim='\t', header=True);")

    # Combined V3
    con.execute(f"""
        CREATE TABLE v3_{target} AS
        SELECT DISTINCT source1_entity_id, matched_entity_id FROM (
            SELECT source1_entity_id, matched_entity_id FROM v2_{target}
            UNION ALL
            SELECT source1_entity_id, matched_entity_id FROM exp1_{target} WHERE s1_fan <= 100
            UNION ALL
            SELECT source1_entity_id, matched_entity_id FROM exp2_{target}
            UNION ALL
            SELECT source1_entity_id, matched_entity_id FROM exp3_{target}
            UNION ALL
            SELECT source1_entity_id, matched_entity_id FROM exp4_{target}
        );
    """)

    # Attribution
    v2_cap = con.execute(f"SELECT count(*) FROM v2_{target} c JOIN gt ON c.source1_entity_id = gt.source1_entity_id AND c.matched_entity_id = gt.target_entity_id WHERE gt.target_source = '{target}'").fetchone()[0]

    con.execute(f"""
        CREATE TABLE e1_filt_{target} AS
        SELECT source1_entity_id, matched_entity_id FROM exp1_{target} WHERE s1_fan <= 100;
    """)

    # E1 new = in E1, not in V2
    e1_new = con.execute(f"""
        SELECT count(*) FROM e1_filt_{target} e
        LEFT JOIN v2_{target} v ON e.source1_entity_id = v.source1_entity_id AND e.matched_entity_id = v.matched_entity_id
        JOIN gt ON e.source1_entity_id = gt.source1_entity_id AND e.matched_entity_id = gt.target_entity_id
        WHERE v.source1_entity_id IS NULL AND gt.target_source = '{target}';
    """).fetchone()[0]

    # E2 new = in E2, not in V2, not in E1
    e2_new = con.execute(f"""
        SELECT count(*) FROM exp2_{target} e
        LEFT JOIN v2_{target} v ON e.source1_entity_id = v.source1_entity_id AND e.matched_entity_id = v.matched_entity_id
        LEFT JOIN e1_filt_{target} e1 ON e.source1_entity_id = e1.source1_entity_id AND e.matched_entity_id = e1.matched_entity_id
        JOIN gt ON e.source1_entity_id = gt.source1_entity_id AND e.matched_entity_id = gt.target_entity_id
        WHERE v.source1_entity_id IS NULL AND e1.source1_entity_id IS NULL AND gt.target_source = '{target}';
    """).fetchone()[0]

    # E3 new = in E3, not in V2, not in E1, not in E2
    e3_new = con.execute(f"""
        SELECT count(*) FROM exp3_{target} e
        LEFT JOIN v2_{target} v ON e.source1_entity_id = v.source1_entity_id AND e.matched_entity_id = v.matched_entity_id
        LEFT JOIN e1_filt_{target} e1 ON e.source1_entity_id = e1.source1_entity_id AND e.matched_entity_id = e1.matched_entity_id
        LEFT JOIN exp2_{target} e2 ON e.source1_entity_id = e2.source1_entity_id AND e.matched_entity_id = e2.matched_entity_id
        JOIN gt ON e.source1_entity_id = gt.source1_entity_id AND e.matched_entity_id = gt.target_entity_id
        WHERE v.source1_entity_id IS NULL AND e1.source1_entity_id IS NULL AND e2.source1_entity_id IS NULL AND gt.target_source = '{target}';
    """).fetchone()[0]

    # E4 new = in E4, not in V2, not in E1, not in E2, not in E3
    e4_new = con.execute(f"""
        SELECT count(*) FROM exp4_{target} e
        LEFT JOIN v2_{target} v ON e.source1_entity_id = v.source1_entity_id AND e.matched_entity_id = v.matched_entity_id
        LEFT JOIN e1_filt_{target} e1 ON e.source1_entity_id = e1.source1_entity_id AND e.matched_entity_id = e1.matched_entity_id
        LEFT JOIN exp2_{target} e2 ON e.source1_entity_id = e2.source1_entity_id AND e.matched_entity_id = e2.matched_entity_id
        LEFT JOIN exp3_{target} e3 ON e.source1_entity_id = e3.source1_entity_id AND e.matched_entity_id = e3.matched_entity_id
        JOIN gt ON e.source1_entity_id = gt.source1_entity_id AND e.matched_entity_id = gt.target_entity_id
        WHERE v.source1_entity_id IS NULL AND e1.source1_entity_id IS NULL AND e2.source1_entity_id IS NULL AND e3.source1_entity_id IS NULL AND gt.target_source = '{target}';
    """).fetchone()[0]

    total_cands = con.execute(f"SELECT count(*) FROM v3_{target}").fetchone()[0]
    total_cap = con.execute(f"SELECT count(*) FROM v3_{target} c JOIN gt ON c.source1_entity_id = gt.source1_entity_id AND c.matched_entity_id = gt.target_entity_id WHERE gt.target_source = '{target}'").fetchone()[0]

    attr[target] = {
        'total_cands': total_cands,
        'v2_cap': v2_cap,
        'e1_new': e1_new,
        'e2_new': e2_new,
        'e3_new': e3_new,
        'e4_new': e4_new,
        'total_cap': total_cap,
        'sum_check': (v2_cap + e1_new + e2_new + e3_new + e4_new) == total_cap
    }

print(json.dumps(attr, indent=2))
with open("P1/reports/v3_incremental_attribution.json", "w") as f:
    json.dump(attr, f, indent=2)
