#!/usr/bin/env python3
"""
P1/scripts/retrieval/attribute_test_fanout.py

Component-level fanout attribution on the canonical TEST split (N=1,732,544).
Analyzes candidate contributions from V2, EXP-01, EXP-02, EXP-03, and EXP-04.
"""

import time
import json
from pathlib import Path
import duckdb

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent

COMMON_PREFIXES = "('the', 'shri', 'sri', 'dr', 'm/s', 'hotel', 'new', 'om', 'sai', 'jai', 'a', 'an')"

DEVANAGARI_MAP = {
    'अ': 'a', 'आ': 'a', 'इ': 'i', 'ई': 'i', 'उ': 'u', 'ऊ': 'u',
    'ऋ': 'r', 'ए': 'e', 'ऐ': 'ai', 'ओ': 'o', 'औ': 'au',
    'क': 'k', 'ख': 'kh', 'ग': 'g', 'घ': 'gh', 'ङ': 'ng',
    'च': 'ch', 'छ': 'chh', 'ज': 'j', 'झ': 'jh', 'ञ': 'ny',
    'ट': 't', 'ठ': 'th', 'ड': 'd', 'ढ': 'dh', 'ण': 'n',
    'त': 't', 'थ': 'th', 'द': 'd', 'ध': 'dh', 'न': 'n',
    'प': 'p', 'फ': 'ph', 'ब': 'b', 'भ': 'bh', 'म': 'm',
    'य': 'y', 'र': 'r', 'ल': 'l', 'व': 'v', 'श': 'sh',
    'ष': 'sh', 'स': 's', 'ह': 'h',
    'ा': 'a', 'ि': 'i', 'ी': 'i', 'ु': 'u', 'ू': 'u',
    'ृ': 'r', 'े': 'e', 'ै': 'ai', 'ो': 'o', 'ौ': 'au',
    '्': '', 'ं': 'n', 'ँ': 'n', 'ः': 'h',
    '०': '0', '१': '1', '२': '2', '३': '3', '४': '4',
    '५': '5', '६': '6', '७': '7', '८': '8', '९': '9'
}

def transliterate_text(text: str) -> str:
    if not text:
        return ""
    res = []
    for char in text:
        res.append(DEVANAGARI_MAP.get(char, char))
    return "".join(res)

def main():
    con = duckdb.connect()
    con.execute("PRAGMA threads=8;")
    con.execute("PRAGMA memory_limit='14GB';")
    con.create_function("transliterate_devanagari", transliterate_text, [str], str)

    print("1. Loading test entities...")
    s1_path = REPO_ROOT / "P1/data/entities/test/source1/test_s1_entities.parquet"
    con.execute(f"""
        CREATE TABLE test_s1 AS
        SELECT
            entity_id,
            country_raw AS country,
            business_name_raw AS name_raw,
            business_address_raw AS addr_raw,
            regexp_replace(lower(trim(business_name_raw)), '[^a-z0-9]', '', 'g') AS clean_name,
            regexp_replace(
                regexp_replace(lower(trim(business_name_raw)), '[^a-z0-9]', '', 'g'),
                '(privatelimited|pvtlimited|pvtltd|pvtld|limited|ltd|llc|inc|corp|corporation|enterprises?|services?)$',
                ''
            ) AS core_name,
            LEFT(regexp_replace(lower(trim(business_name_raw)), '[^a-z0-9]', '', 'g'), 2) AS name_prefix_2,
            LEFT(regexp_replace(lower(trim(business_name_raw)), '[^a-z0-9]', '', 'g'), 3) AS name_prefix_3,
            regexp_replace(regexp_extract(trim(business_address_raw), '[0-9]+[A-Za-z]?', 0), '^0+', '') AS house_norm,
            regexp_replace(lower(trim(business_address_raw)), '[^a-z0-9]', '', 'g') AS addr_clean,
            regexp_extract(trim(business_address_raw), '(?:^|[^0-9])([0-9]{{5,6}})(?:[^0-9]|$)', 1) AS postal_code,
            CASE
                WHEN lower(split_part(trim(business_name_raw), ' ', 1)) IN {COMMON_PREFIXES}
                     AND split_part(trim(business_name_raw), ' ', 2) <> ''
                THEN split_part(trim(business_name_raw), ' ', 2)
                ELSE split_part(trim(business_name_raw), ' ', 1)
            END AS root_token,
            LEFT(regexp_replace(lower(trim(transliterate_devanagari(business_name_raw))), '[^a-z0-9]', '', 'g'), 2) AS translit_prefix_2,
            regexp_replace(lower(trim(transliterate_devanagari(business_name_raw))), '[^a-z0-9]', '', 'g') AS translit_name
        FROM read_parquet('{s1_path}');
    """)
    n_s1 = con.execute("SELECT count(*) FROM test_s1").fetchone()[0]
    print(f"Total Test S1: {n_s1:,}")

    print("2. Generating test components for S2 and S3...")
    for target in ['s2', 's3']:
        tgt_parquet = REPO_ROOT / f"P1/data/entities/test/{'source2' if target == 's2' else 'source3'}/test_{target}_entities.parquet"
        con.execute(f"""
            CREATE TABLE test_{target} AS
            SELECT
                entity_id,
                country_raw AS country,
                business_name_raw AS name_raw,
                business_address_raw AS addr_raw,
                regexp_replace(lower(trim(business_name_raw)), '[^a-z0-9]', '', 'g') AS clean_name,
                regexp_replace(
                    regexp_replace(lower(trim(business_name_raw)), '[^a-z0-9]', '', 'g'),
                    '(privatelimited|pvtlimited|pvtltd|pvtld|limited|ltd|llc|inc|corp|corporation|enterprises?|services?)$',
                    ''
                ) AS core_name,
                LEFT(regexp_replace(lower(trim(business_name_raw)), '[^a-z0-9]', '', 'g'), 2) AS name_prefix_2,
                LEFT(regexp_replace(lower(trim(business_name_raw)), '[^a-z0-9]', '', 'g'), 3) AS name_prefix_3,
                regexp_replace(regexp_extract(trim(business_address_raw), '[0-9]+[A-Za-z]?', 0), '^0+', '') AS house_norm,
                regexp_replace(lower(trim(business_address_raw)), '[^a-z0-9]', '', 'g') AS addr_clean,
                regexp_extract(trim(business_address_raw), '(?:^|[^0-9])([0-9]{{5,6}})(?:[^0-9]|$)', 1) AS postal_code,
                CASE
                    WHEN lower(split_part(trim(business_name_raw), ' ', 1)) IN {COMMON_PREFIXES}
                         AND split_part(trim(business_name_raw), ' ', 2) <> ''
                    THEN split_part(trim(business_name_raw), ' ', 2)
                    ELSE split_part(trim(business_name_raw), ' ', 1)
                END AS root_token,
                LEFT(regexp_replace(lower(trim(transliterate_devanagari(business_name_raw))), '[^a-z0-9]', '', 'g'), 2) AS translit_prefix_2,
                regexp_replace(lower(trim(transliterate_devanagari(business_name_raw))), '[^a-z0-9]', '', 'g') AS translit_name
            FROM read_parquet('{tgt_parquet}');
        """)

        # V2
        v2_path = REPO_ROOT / f"P2/data/candidates/test_candidate_pairs_{target}_v2.tsv"
        con.execute(f"CREATE TABLE test_v2_{target} AS SELECT source1_entity_id, matched_entity_id FROM read_csv_auto('{v2_path}', delim='\t', header=True);")

        # EXP-01 (cap <= 100)
        con.execute(f"""
            CREATE TEMP TABLE test_e1_raw_{target} AS
            SELECT s1.entity_id AS source1_entity_id, tgt.entity_id AS matched_entity_id
            FROM test_s1 s1 JOIN test_{target} tgt ON s1.country = tgt.country AND s1.core_name = tgt.core_name
            WHERE LENGTH(s1.core_name) >= 5;
        """)
        con.execute(f"""
            CREATE TABLE test_e1_{target} AS
            SELECT r.source1_entity_id, r.matched_entity_id
            FROM test_e1_raw_{target} r
            JOIN (SELECT source1_entity_id FROM test_e1_raw_{target} GROUP BY source1_entity_id HAVING count(*) <= 100) k
              ON r.source1_entity_id = k.source1_entity_id;
        """)

        # EXP-02
        con.execute(f"""
            CREATE TABLE test_e2_{target} AS
            SELECT s1.entity_id AS source1_entity_id, tgt.entity_id AS matched_entity_id
            FROM test_s1 s1 JOIN test_{target} tgt
              ON s1.country = tgt.country AND s1.house_norm = tgt.house_norm AND s1.name_prefix_2 = tgt.name_prefix_2
            WHERE s1.house_norm IS NOT NULL AND s1.house_norm <> ''
              AND LENGTH(s1.clean_name) >= 2 AND LENGTH(tgt.clean_name) >= 2
              AND jaro_winkler_similarity(s1.clean_name, tgt.clean_name) >= 0.80;
        """)

        # EXP-03
        con.execute(f"""
            CREATE TABLE test_e3_{target} AS
            SELECT DISTINCT source1_entity_id, matched_entity_id FROM (
                SELECT s1.entity_id AS source1_entity_id, tgt.entity_id AS matched_entity_id
                FROM test_s1 s1 JOIN test_{target} tgt
                  ON s1.country = tgt.country AND s1.house_norm = tgt.house_norm AND s1.translit_prefix_2 = tgt.translit_prefix_2
                WHERE s1.house_norm IS NOT NULL AND s1.house_norm <> ''
                  AND LENGTH(s1.translit_name) >= 2 AND LENGTH(tgt.translit_name) >= 2
                  AND jaro_winkler_similarity(s1.translit_name, tgt.translit_name) >= 0.80
                UNION ALL
                SELECT s1.entity_id AS source1_entity_id, tgt.entity_id AS matched_entity_id
                FROM test_s1 s1 JOIN test_{target} tgt
                  ON s1.country = tgt.country AND s1.house_norm = tgt.house_norm AND s1.postal_code = tgt.postal_code
                WHERE s1.country = 'India' AND s1.house_norm IS NOT NULL AND s1.house_norm <> ''
                  AND s1.postal_code IS NOT NULL AND LENGTH(s1.postal_code) >= 5
            );
        """)

        # EXP-04
        con.execute(f"""
            CREATE TABLE test_e4_{target} AS
            SELECT DISTINCT source1_entity_id, matched_entity_id FROM (
                SELECT s1.entity_id AS source1_entity_id, tgt.entity_id AS matched_entity_id
                FROM test_s1 s1 JOIN test_{target} tgt
                  ON s1.country = tgt.country AND LEFT(s1.addr_clean, 10) = LEFT(tgt.addr_clean, 10) AND s1.name_prefix_3 = tgt.name_prefix_3
                WHERE LENGTH(s1.addr_clean) >= 10 AND LENGTH(tgt.addr_clean) >= 10 AND LENGTH(s1.name_prefix_3) = 3
                UNION ALL
                SELECT s1.entity_id AS source1_entity_id, tgt.entity_id AS matched_entity_id
                FROM test_s1 s1 JOIN test_{target} tgt
                  ON s1.country = tgt.country AND s1.postal_code = tgt.postal_code AND s1.root_token = tgt.root_token
                WHERE s1.postal_code IS NOT NULL AND LENGTH(s1.postal_code) >= 5 AND s1.root_token IS NOT NULL AND LENGTH(s1.root_token) >= 4
            );
        """)

    print("\n3. Evaluating configurations on TEST...")
    configs = {
        "A. V2 only": ["test_v2_s2", "test_v2_s3"],
        "B. V2 + EXP-RET-01": ["test_v2_s2", "test_v2_s3", "test_e1_s2", "test_e1_s3"],
        "C. V2 + EXP-RET-02": ["test_v2_s2", "test_v2_s3", "test_e2_s2", "test_e2_s3"],
        "D. V2 + EXP-RET-03": ["test_v2_s2", "test_v2_s3", "test_e3_s2", "test_e3_s3"],
        "E. V2 + EXP-RET-04": ["test_v2_s2", "test_v2_s3", "test_e4_s2", "test_e4_s3"],
        "F. V2 + EXP-02 + EXP-03 + EXP-04": ["test_v2_s2", "test_v2_s3", "test_e2_s2", "test_e2_s3", "test_e3_s2", "test_e3_s3", "test_e4_s2", "test_e4_s3"],
        "G. Full V3 (V2+01+02+03+04)": ["test_v2_s2", "test_v2_s3", "test_e1_s2", "test_e1_s3", "test_e2_s2", "test_e2_s3", "test_e3_s2", "test_e3_s3", "test_e4_s2", "test_e4_s3"],
        "H. V2 + EXP-01 + EXP-04": ["test_v2_s2", "test_v2_s3", "test_e1_s2", "test_e1_s3", "test_e4_s2", "test_e4_s3"],
        "I. V2 + EXP-01 + EXP-02": ["test_v2_s2", "test_v2_s3", "test_e1_s2", "test_e1_s3", "test_e2_s2", "test_e2_s3"],
        "J. V2 + EXP-01 + EXP-03": ["test_v2_s2", "test_v2_s3", "test_e1_s2", "test_e1_s3", "test_e3_s2", "test_e3_s3"]
    }

    results = []
    for name, tbls in configs.items():
        t0 = time.time()
        union_sql = " UNION ALL ".join([f"SELECT source1_entity_id, matched_entity_id FROM {t}" for t in tbls])
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE cfg_cands AS
            SELECT DISTINCT source1_entity_id, matched_entity_id FROM ({union_sql});
        """)
        c_count = con.execute("SELECT count(*) FROM cfg_cands").fetchone()[0]

        con.execute("""
            CREATE OR REPLACE TEMP TABLE cfg_fanout AS
            SELECT s.entity_id as s1_id, COALESCE(c.cnt, 0) as cnt
            FROM test_s1 s
            LEFT JOIN (SELECT source1_entity_id, count(*) as cnt FROM cfg_cands GROUP BY source1_entity_id) c
              ON s.entity_id = c.source1_entity_id;
        """)

        stats = con.execute("""
            SELECT
                count(CASE WHEN cnt = 0 THEN 1 END),
                quantile_cont(cnt, 0.50),
                quantile_cont(cnt, 0.90),
                quantile_cont(cnt, 0.95),
                quantile_cont(cnt, 0.99),
                quantile_cont(cnt, 0.999),
                max(cnt),
                count(CASE WHEN cnt >= 300 THEN 1 END),
                count(CASE WHEN cnt >= 500 THEN 1 END),
                count(CASE WHEN cnt >= 1000 THEN 1 END)
            FROM cfg_fanout;
        """).fetchone()

        row = {
            "config": name,
            "candidates": c_count,
            "zero_s1": stats[0],
            "p50": stats[1],
            "p90": stats[2],
            "p95": stats[3],
            "p99": stats[4],
            "p999": stats[5],
            "max": stats[6],
            "ge_300": stats[7],
            "ge_500": stats[8],
            "ge_1000": stats[9]
        }
        results.append(row)
        print(f"Computed {name} in {time.time()-t0:.2f}s: Cands={c_count:,}, p95={stats[3]:.1f}, >=300={stats[7]:,}, Zero-S1={stats[0]:,}")

    print("\n4. Upper-Tail Analysis on Full V3...")
    con.execute("""
        CREATE OR REPLACE TABLE full_v3_fanout AS
        SELECT s.entity_id as s1_id, s.country, s.clean_name, COALESCE(c.cnt, 0) as total_fan
        FROM test_s1 s
        LEFT JOIN (
            SELECT source1_entity_id, count(*) as cnt
            FROM (
                SELECT DISTINCT source1_entity_id, matched_entity_id
                FROM (
                    SELECT * FROM test_v2_s2 UNION ALL SELECT * FROM test_v2_s3
                    UNION ALL SELECT * FROM test_e1_s2 UNION ALL SELECT * FROM test_e1_s3
                    UNION ALL SELECT * FROM test_e2_s2 UNION ALL SELECT * FROM test_e2_s3
                    UNION ALL SELECT * FROM test_e3_s2 UNION ALL SELECT * FROM test_e3_s3
                    UNION ALL SELECT * FROM test_e4_s2 UNION ALL SELECT * FROM test_e4_s3
                )
            ) GROUP BY source1_entity_id
        ) c ON s.entity_id = c.source1_entity_id;
    """)

    n_ge_300 = con.execute("SELECT count(*) FROM full_v3_fanout WHERE total_fan >= 300").fetchone()[0]
    n_ge_500 = con.execute("SELECT count(*) FROM full_v3_fanout WHERE total_fan >= 500").fetchone()[0]
    n_ge_1000 = con.execute("SELECT count(*) FROM full_v3_fanout WHERE total_fan >= 1000").fetchone()[0]

    print(f"Entities with fanout >= 300: {n_ge_300:,} ({n_ge_300/n_s1:.4%})")
    print(f"Entities with fanout >= 500: {n_ge_500:,} ({n_ge_500/n_s1:.4%})")
    print(f"Entities with fanout >= 1000: {n_ge_1000:,} ({n_ge_1000/n_s1:.4%})")

    # Sample breakdown of top 10 highest fanout entities by rule
    top10 = con.execute("SELECT s1_id, total_fan FROM full_v3_fanout ORDER BY total_fan DESC LIMIT 10").fetchall()
    print("\nTop 10 High-Fanout Entities Breakdown by Rule:")
    breakdowns = []
    for s1_id, tot in top10:
        v2_c = con.execute(f"SELECT count(DISTINCT matched_entity_id) FROM (SELECT matched_entity_id FROM test_v2_s2 WHERE source1_entity_id = '{s1_id}' UNION ALL SELECT matched_entity_id FROM test_v2_s3 WHERE source1_entity_id = '{s1_id}')").fetchone()[0]
        e1_c = con.execute(f"SELECT count(DISTINCT matched_entity_id) FROM (SELECT matched_entity_id FROM test_e1_s2 WHERE source1_entity_id = '{s1_id}' UNION ALL SELECT matched_entity_id FROM test_e1_s3 WHERE source1_entity_id = '{s1_id}')").fetchone()[0]
        e2_c = con.execute(f"SELECT count(DISTINCT matched_entity_id) FROM (SELECT matched_entity_id FROM test_e2_s2 WHERE source1_entity_id = '{s1_id}' UNION ALL SELECT matched_entity_id FROM test_e2_s3 WHERE source1_entity_id = '{s1_id}')").fetchone()[0]
        e3_c = con.execute(f"SELECT count(DISTINCT matched_entity_id) FROM (SELECT matched_entity_id FROM test_e3_s2 WHERE source1_entity_id = '{s1_id}' UNION ALL SELECT matched_entity_id FROM test_e3_s3 WHERE source1_entity_id = '{s1_id}')").fetchone()[0]
        e4_c = con.execute(f"SELECT count(DISTINCT matched_entity_id) FROM (SELECT matched_entity_id FROM test_e4_s2 WHERE source1_entity_id = '{s1_id}' UNION ALL SELECT matched_entity_id FROM test_e4_s3 WHERE source1_entity_id = '{s1_id}')").fetchone()[0]
        breakdowns.append({
            "s1_id": s1_id,
            "total_fan": tot,
            "v2": v2_c,
            "exp01": e1_c,
            "exp02": e2_c,
            "exp03": e3_c,
            "exp04": e4_c
        })
        print(f"  {s1_id}: Total={tot:,} | V2={v2_c} | EXP-01={e1_c} | EXP-02={e2_c} | EXP-03={e3_c} | EXP-04={e4_c}")

    out_json = {
        "configurations": results,
        "upper_tail": {
            "ge_300": n_ge_300,
            "ge_500": n_ge_500,
            "ge_1000": n_ge_1000
        },
        "top_entities": breakdowns
    }

    with open(REPO_ROOT / "P1/reports/test_fanout_attribution.json", "w") as f:
        json.dump(out_json, f, indent=2)
    print("\nSaved report to P1/reports/test_fanout_attribution.json")

if __name__ == "__main__":
    main()
