#!/usr/bin/env python3
"""
P1/scripts/retrieval/generate_v3_candidates.py

Authoritative candidate generation pipeline V3 for Amazon ML Challenge 2026.
Combines:
  - Frozen V2 baseline candidate rules (Rules A through I)
  - EXP-RET-01: Soft Name Retrieval (Country + core_name len >= 5 with S1 fanout <= 100)
  - EXP-RET-02: Relaxed Name + House Retrieval (Country + house_norm + prefix2 + Jaro-Winkler >= 0.80)
  - EXP-RET-03: Indic / Transliteration-Aware Retrieval (Devanagari ISO 15919 transliteration + house_norm + transliterated prefix2 JW >= 0.80 + Cross-script address block)
  - EXP-RET-04: Locality + Name Retrieval (Country + address prefix 10 + name prefix 3 + Country + postal_code len >= 5 + root_token len >= 4)

Produces standardized, versioned V3 candidate artifacts for Train and Test:
  - P1/data/candidates/v3/train_candidate_pairs_s2_v3.tsv
  - P1/data/candidates/v3/train_candidate_pairs_s3_v3.tsv
  - P1/data/candidates/v3/test_candidate_pairs_s2_v3.tsv
  - P1/data/candidates/v3/test_candidate_pairs_s3_v3.tsv
"""

import os
import sys
import argparse
import duckdb
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
OUTPUT_DIR = REPO_ROOT / "P1" / "data" / "candidates" / "v3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
        if char in DEVANAGARI_MAP:
            res.append(DEVANAGARI_MAP[char])
        else:
            res.append(char)
    return "".join(res)

def generate_train_candidates(con: duckdb.DuckDBPyConnection):
    """Generate V3 candidates for Train split using precomputed and validated component sets."""
    print("\n--- Generating V3 Candidates for TRAIN ---")
    for target in ['s2', 's3']:
        t0 = time.time()
        print(f"Generating train_candidate_pairs_{target}_v3.tsv...")
        v2_path = REPO_ROOT / f"P2/data/candidates/train_candidate_pairs_{target}_v2.tsv"
        exp1_path = REPO_ROOT / f"P1/experiments/phase3/EXP-RET-01/train_candidates_{target}_exp_ret_01.tsv"
        exp2_path = REPO_ROOT / f"P1/experiments/phase3/EXP-RET-02/train_candidates_{target}_exp_ret_02_th080.tsv"
        exp3_path = REPO_ROOT / f"P1/experiments/phase3/EXP-RET-03/train_candidates_{target}_exp_ret_03.tsv"
        exp4_path = REPO_ROOT / f"P1/experiments/phase3/EXP-RET-04/train_candidates_{target}_exp_ret_04.tsv"
        out_path = OUTPUT_DIR / f"train_candidate_pairs_{target}_v3.tsv"

        con.execute(f"CREATE OR REPLACE TABLE v2_{target} AS SELECT source1_entity_id, matched_entity_id FROM read_csv_auto('{v2_path}', delim='\t', header=True);")

        # EXP-RET-01 with validated cap <= 100
        con.execute(f"""
            CREATE TEMP TABLE exp1_raw_{target} AS SELECT source1_entity_id, matched_entity_id FROM read_csv_auto('{exp1_path}', delim='\t', header=True);
            CREATE OR REPLACE TABLE exp1_{target} AS
            SELECT r.source1_entity_id, r.matched_entity_id
            FROM exp1_raw_{target} r
            JOIN (SELECT source1_entity_id FROM exp1_raw_{target} GROUP BY source1_entity_id HAVING count(*) <= 100) k
              ON r.source1_entity_id = k.source1_entity_id;
            DROP TABLE exp1_raw_{target};
        """)

        # EXP-RET-02 with validated cap <= 50
        con.execute(f"""
            CREATE TEMP TABLE exp2_raw_{target} AS SELECT source1_entity_id, matched_entity_id FROM read_csv_auto('{exp2_path}', delim='\t', header=True);
            CREATE OR REPLACE TABLE exp2_{target} AS
            SELECT r.source1_entity_id, r.matched_entity_id
            FROM exp2_raw_{target} r
            JOIN (SELECT source1_entity_id FROM exp2_raw_{target} GROUP BY source1_entity_id HAVING count(*) <= 50) k
              ON r.source1_entity_id = k.source1_entity_id;
            DROP TABLE exp2_raw_{target};
        """)

        # EXP-RET-03 with validated cap <= 50
        con.execute(f"""
            CREATE TEMP TABLE exp3_raw_{target} AS SELECT source1_entity_id, matched_entity_id FROM read_csv_auto('{exp3_path}', delim='\t', header=True);
            CREATE OR REPLACE TABLE exp3_{target} AS
            SELECT r.source1_entity_id, r.matched_entity_id
            FROM exp3_raw_{target} r
            JOIN (SELECT source1_entity_id FROM exp3_raw_{target} GROUP BY source1_entity_id HAVING count(*) <= 50) k
              ON r.source1_entity_id = k.source1_entity_id;
            DROP TABLE exp3_raw_{target};
        """)

        con.execute(f"CREATE OR REPLACE TABLE exp4_{target} AS SELECT source1_entity_id, matched_entity_id FROM read_csv_auto('{exp4_path}', delim='\t', header=True);")

        con.execute(f"""
            CREATE OR REPLACE TABLE v3_train_{target} AS
            SELECT DISTINCT source1_entity_id, matched_entity_id
            FROM (
                SELECT source1_entity_id, matched_entity_id FROM v2_{target}
                UNION ALL
                SELECT source1_entity_id, matched_entity_id FROM exp1_{target}
                UNION ALL
                SELECT source1_entity_id, matched_entity_id FROM exp2_{target}
                UNION ALL
                SELECT source1_entity_id, matched_entity_id FROM exp3_{target}
                UNION ALL
                SELECT source1_entity_id, matched_entity_id FROM exp4_{target}
            )
            ORDER BY source1_entity_id, matched_entity_id;
        """)

        count = con.execute(f"SELECT count(*) FROM v3_train_{target}").fetchone()[0]
        print(f"Exporting {count:,} candidate pairs to {out_path}...")
        con.execute(f"COPY v3_train_{target} TO '{out_path}' (HEADER, DELIMITER '\t');")
        print(f"Done in {time.time()-t0:.2f}s: {out_path}")

def generate_test_candidates(con: duckdb.DuckDBPyConnection):
    """Generate V3 candidates for Test split applying the exact same rules symmetrically."""
    print("\n--- Generating V3 Candidates for TEST ---")
    con.create_function("transliterate_devanagari", transliterate_text, [str], str)

    # 1. Load canonical test S1 entities
    print("Loading test S1 entities...")
    s1_path = REPO_ROOT / "P1/data/entities/test/source1/test_s1_entities.parquet"
    con.execute(f"""
        CREATE OR REPLACE TABLE test_s1 AS
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
    s1_cnt = con.execute("SELECT count(*) FROM test_s1").fetchone()[0]
    print(f"Loaded {s1_cnt:,} test S1 entities.")

    for target in ['s2', 's3']:
        t0 = time.time()
        print(f"\nProcessing test {target.upper()}...")
        tgt_parquet = REPO_ROOT / f"P1/data/entities/test/{'source2' if target == 's2' else 'source3'}/test_{target}_entities.parquet"
        con.execute(f"""
            CREATE OR REPLACE TABLE test_{target} AS
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

        # 1. Baseline V2 test candidates
        v2_test_path = REPO_ROOT / f"P2/data/candidates/test_candidate_pairs_{target}_v2.tsv"
        print(f"Loading V2 test candidates from {v2_test_path}...")
        con.execute(f"""
            CREATE OR REPLACE TABLE test_v2_{target} AS
            SELECT source1_entity_id, matched_entity_id FROM read_csv_auto('{v2_test_path}', delim='\t', header=True);
        """)

        # 2. EXP-RET-01 on Test (core_name len >= 5 with s1_fan <= 100)
        print("Generating test EXP-RET-01 candidates...")
        con.execute(f"""
            CREATE OR REPLACE TABLE test_exp1_{target} AS
            SELECT s1.entity_id AS source1_entity_id, tgt.entity_id AS matched_entity_id
            FROM test_s1 s1
            JOIN test_{target} tgt
              ON s1.country = tgt.country
             AND s1.core_name = tgt.core_name
            WHERE LENGTH(s1.core_name) >= 5;
        """)

        # Filter EXP-01 by fanout <= 100 per S1
        con.execute(f"""
            CREATE OR REPLACE TABLE test_exp1_filt_{target} AS
            SELECT r.source1_entity_id, r.matched_entity_id
            FROM test_exp1_{target} r
            JOIN (SELECT source1_entity_id FROM test_exp1_{target} GROUP BY source1_entity_id HAVING count(*) <= 100) k
              ON r.source1_entity_id = k.source1_entity_id;
            DROP TABLE test_exp1_{target};
        """)

        # 3. EXP-RET-02 on Test (house_norm + prefix2 with JW >= 0.80) with validated cap <= 50
        print("Generating test EXP-RET-02 candidates...")
        con.execute(f"""
            CREATE TEMP TABLE test_exp2_raw_{target} AS
            SELECT s1.entity_id AS source1_entity_id, tgt.entity_id AS matched_entity_id
            FROM test_s1 s1
            JOIN test_{target} tgt
              ON s1.country = tgt.country
             AND s1.house_norm = tgt.house_norm
             AND s1.name_prefix_2 = tgt.name_prefix_2
            WHERE s1.house_norm IS NOT NULL AND s1.house_norm <> ''
              AND LENGTH(s1.clean_name) >= 2 AND LENGTH(tgt.clean_name) >= 2
              AND jaro_winkler_similarity(s1.clean_name, tgt.clean_name) >= 0.80;

            CREATE OR REPLACE TABLE test_exp2_{target} AS
            SELECT r.source1_entity_id, r.matched_entity_id
            FROM test_exp2_raw_{target} r
            JOIN (SELECT source1_entity_id FROM test_exp2_raw_{target} GROUP BY source1_entity_id HAVING count(*) <= 50) k
              ON r.source1_entity_id = k.source1_entity_id;
            DROP TABLE test_exp2_raw_{target};
        """)

        # 4. EXP-RET-03 on Test (Indic transliteration + cross-script address) with validated cap <= 50
        print("Generating test EXP-RET-03 candidates...")
        con.execute(f"""
            CREATE TEMP TABLE test_exp3_raw_{target} AS
            SELECT DISTINCT source1_entity_id, matched_entity_id FROM (
                SELECT s1.entity_id AS source1_entity_id, tgt.entity_id AS matched_entity_id
                FROM test_s1 s1
                JOIN test_{target} tgt
                  ON s1.country = tgt.country
                 AND s1.house_norm = tgt.house_norm
                 AND s1.translit_prefix_2 = tgt.translit_prefix_2
                WHERE s1.house_norm IS NOT NULL AND s1.house_norm <> ''
                  AND LENGTH(s1.translit_name) >= 2 AND LENGTH(tgt.translit_name) >= 2
                  AND jaro_winkler_similarity(s1.translit_name, tgt.translit_name) >= 0.80
                UNION ALL
                SELECT s1.entity_id AS source1_entity_id, tgt.entity_id AS matched_entity_id
                FROM test_s1 s1
                JOIN test_{target} tgt
                  ON s1.country = tgt.country
                 AND s1.house_norm = tgt.house_norm
                 AND s1.postal_code = tgt.postal_code
                WHERE s1.country = 'India'
                  AND s1.house_norm IS NOT NULL AND s1.house_norm <> ''
                  AND s1.postal_code IS NOT NULL AND LENGTH(s1.postal_code) >= 5
            );

            CREATE OR REPLACE TABLE test_exp3_{target} AS
            SELECT r.source1_entity_id, r.matched_entity_id
            FROM test_exp3_raw_{target} r
            JOIN (SELECT source1_entity_id FROM test_exp3_raw_{target} GROUP BY source1_entity_id HAVING count(*) <= 50) k
              ON r.source1_entity_id = k.source1_entity_id;
            DROP TABLE test_exp3_raw_{target};
        """)

        # 5. EXP-RET-04 on Test (Locality address shingle 10 + name_prefix_3 + Postal len>=5 + root_token len>=4)
        print("Generating test EXP-RET-04 candidates...")
        con.execute(f"""
            CREATE OR REPLACE TABLE test_exp4_{target} AS
            SELECT DISTINCT source1_entity_id, matched_entity_id FROM (
                SELECT s1.entity_id AS source1_entity_id, tgt.entity_id AS matched_entity_id
                FROM test_s1 s1
                JOIN test_{target} tgt
                  ON s1.country = tgt.country
                 AND LEFT(s1.addr_clean, 10) = LEFT(tgt.addr_clean, 10)
                 AND s1.name_prefix_3 = tgt.name_prefix_3
                WHERE LENGTH(s1.addr_clean) >= 10 AND LENGTH(tgt.addr_clean) >= 10
                  AND s1.name_prefix_3 IS NOT NULL AND LENGTH(s1.name_prefix_3) = 3
                UNION ALL
                SELECT s1.entity_id AS source1_entity_id, tgt.entity_id AS matched_entity_id
                FROM test_s1 s1
                JOIN test_{target} tgt
                  ON s1.country = tgt.country
                 AND s1.postal_code = tgt.postal_code
                 AND s1.root_token = tgt.root_token
                WHERE s1.postal_code IS NOT NULL AND LENGTH(s1.postal_code) >= 5
                  AND s1.root_token IS NOT NULL AND LENGTH(s1.root_token) >= 4
            );
        """)

        # Combine all for Test
        out_test_path = OUTPUT_DIR / f"test_candidate_pairs_{target}_v3.tsv"
        print(f"Combining and exporting V3 test candidates to {out_test_path}...")
        con.execute(f"""
            CREATE OR REPLACE TABLE v3_test_{target} AS
            SELECT DISTINCT source1_entity_id, matched_entity_id
            FROM (
                SELECT source1_entity_id, matched_entity_id FROM test_v2_{target}
                UNION ALL
                SELECT source1_entity_id, matched_entity_id FROM test_exp1_filt_{target}
                UNION ALL
                SELECT source1_entity_id, matched_entity_id FROM test_exp2_{target}
                UNION ALL
                SELECT source1_entity_id, matched_entity_id FROM test_exp3_{target}
                UNION ALL
                SELECT source1_entity_id, matched_entity_id FROM test_exp4_{target}
            )
            ORDER BY source1_entity_id, matched_entity_id;
        """)

        test_cnt = con.execute(f"SELECT count(*) FROM v3_test_{target}").fetchone()[0]
        con.execute(f"COPY v3_test_{target} TO '{out_test_path}' (HEADER, DELIMITER '\t');")
        print(f"Exported {test_cnt:,} test candidate pairs for {target.upper()} in {time.time()-t0:.2f}s.")

def main():
    parser = argparse.ArgumentParser(description="Generate V3 Candidates")
    parser.add_argument("--split", choices=["train", "test", "all"], default="all")
    args = parser.parse_args()

    if args.split in ["train", "all"]:
        con_train = duckdb.connect()
        con_train.execute("PRAGMA threads=8;")
        con_train.execute("PRAGMA memory_limit='10GB';")
        generate_train_candidates(con_train)
        con_train.close()

    if args.split in ["test", "all"]:
        con_test = duckdb.connect()
        con_test.execute("PRAGMA threads=8;")
        con_test.execute("PRAGMA memory_limit='10GB';")
        generate_test_candidates(con_test)
        con_test.close()

    print("\nV3 Candidate generation completed successfully.")

if __name__ == "__main__":
    main()
