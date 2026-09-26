#!/usr/bin/env python3
"""
Controlled Fanout-Correction Experiment: EXP-RET-01_CAP70
Enforces EXP-RET-01 per-S1 cap = 70 using an optimized two-pass group-by filter.
Generates:
  TRAIN:
    P1/experiments/phase3/EXP-RET-01_CAP70/train_candidate_pairs_s2_cap70.tsv
    P1/experiments/phase3/EXP-RET-01_CAP70/train_candidate_pairs_s3_cap70.tsv
  TEST:
    P1/experiments/phase3/EXP-RET-01_CAP70/test_candidate_pairs_s2_cap70.tsv
    P1/experiments/phase3/EXP-RET-01_CAP70/test_candidate_pairs_s3_cap70.tsv
"""

import os
import sys
import json
import time
import hashlib
from pathlib import Path
import duckdb

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
EXP_DIR = REPO_ROOT / "P1" / "experiments" / "phase3" / "EXP-RET-01_CAP70"
EXP_DIR.mkdir(parents=True, exist_ok=True)

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

def get_file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def main():
    t_global_start = time.time()
    con = duckdb.connect()
    con.execute("PRAGMA threads=8;")
    con.execute("PRAGMA memory_limit='14GB';")
    con.create_function("transliterate_devanagari", transliterate_text, [str], str)
    
    print("=== Step 1: Generating TRAIN Candidates (EXP-01 cap=70) ===")
    for target in ['s2', 's3']:
        t0 = time.time()
        print(f"Generating train candidates for {target.upper()}...")
        v2_path = REPO_ROOT / f"P2/data/candidates/train_candidate_pairs_{target}_v2.tsv"
        exp1_path = REPO_ROOT / f"P1/experiments/phase3/EXP-RET-01/train_candidates_{target}_exp_ret_01.tsv"
        exp2_path = REPO_ROOT / f"P1/experiments/phase3/EXP-RET-02/train_candidates_{target}_exp_ret_02_th080.tsv"
        exp3_path = REPO_ROOT / f"P1/experiments/phase3/EXP-RET-03/train_candidates_{target}_exp_ret_03.tsv"
        exp4_path = REPO_ROOT / f"P1/experiments/phase3/EXP-RET-04/train_candidates_{target}_exp_ret_04.tsv"
        out_path = EXP_DIR / f"train_candidate_pairs_{target}_cap70.tsv"
        
        con.execute(f"CREATE OR REPLACE TABLE v2_{target} AS SELECT source1_entity_id, matched_entity_id FROM read_csv_auto('{v2_path}', delim='\t', header=True);")
        
        # Optimized two-pass group-by cap for EXP-01
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE exp1_valid_{target} AS
            SELECT source1_entity_id
            FROM read_csv_auto('{exp1_path}', delim='\t', header=True)
            GROUP BY source1_entity_id
            HAVING count(*) <= 70;
        """)
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE exp1_filt_{target} AS
            SELECT c.source1_entity_id, c.matched_entity_id
            FROM read_csv_auto('{exp1_path}', delim='\t', header=True) c
            JOIN exp1_valid_{target} v ON c.source1_entity_id = v.source1_entity_id;
        """)
        
        con.execute(f"CREATE OR REPLACE TABLE exp2_{target} AS SELECT source1_entity_id, matched_entity_id FROM read_csv_auto('{exp2_path}', delim='\t', header=True);")
        con.execute(f"CREATE OR REPLACE TABLE exp3_{target} AS SELECT source1_entity_id, matched_entity_id FROM read_csv_auto('{exp3_path}', delim='\t', header=True);")
        con.execute(f"CREATE OR REPLACE TABLE exp4_{target} AS SELECT source1_entity_id, matched_entity_id FROM read_csv_auto('{exp4_path}', delim='\t', header=True);")
        
        con.execute(f"""
            CREATE OR REPLACE TABLE train_{target}_cap70 AS
            SELECT DISTINCT source1_entity_id, matched_entity_id
            FROM (
                SELECT source1_entity_id, matched_entity_id FROM v2_{target}
                UNION ALL
                SELECT source1_entity_id, matched_entity_id FROM exp1_filt_{target}
                UNION ALL
                SELECT source1_entity_id, matched_entity_id FROM exp2_{target}
                UNION ALL
                SELECT source1_entity_id, matched_entity_id FROM exp3_{target}
                UNION ALL
                SELECT source1_entity_id, matched_entity_id FROM exp4_{target}
            )
            ORDER BY source1_entity_id, matched_entity_id;
        """)
        
        cnt = con.execute(f"SELECT count(*) FROM train_{target}_cap70").fetchone()[0]
        print(f"Exporting {cnt:,} candidate pairs to {out_path}...")
        con.execute(f"COPY train_{target}_cap70 TO '{out_path}' (HEADER, DELIMITER '\t');")
        print(f"Exported in {time.time()-t0:.2f}s.")

    print("\n=== Step 2: Generating TEST Candidates (EXP-01 cap=70) ===")
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
    
    for target in ['s2', 's3']:
        t0 = time.time()
        print(f"Processing test {target.upper()} with cap=70...")
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
        
        v2_test_path = REPO_ROOT / f"P2/data/candidates/test_candidate_pairs_{target}_v2.tsv"
        con.execute(f"CREATE OR REPLACE TABLE test_v2_{target} AS SELECT source1_entity_id, matched_entity_id FROM read_csv_auto('{v2_test_path}', delim='\t', header=True);")
        
        # EXP-01 test with optimized two-pass cap <= 70
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE test_exp1_raw_{target} AS
            SELECT s1.entity_id AS source1_entity_id, tgt.entity_id AS matched_entity_id
            FROM test_s1 s1
            JOIN test_{target} tgt 
              ON s1.country = tgt.country 
             AND s1.core_name = tgt.core_name
            WHERE LENGTH(s1.core_name) >= 5;
        """)
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE test_exp1_valid_{target} AS
            SELECT source1_entity_id
            FROM test_exp1_raw_{target}
            GROUP BY source1_entity_id
            HAVING count(*) <= 70;
        """)
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE test_exp1_filt_{target} AS
            SELECT r.source1_entity_id, r.matched_entity_id
            FROM test_exp1_raw_{target} r
            JOIN test_exp1_valid_{target} v ON r.source1_entity_id = v.source1_entity_id;
        """)
        
        # EXP-02 test
        con.execute(f"""
            CREATE OR REPLACE TABLE test_exp2_{target} AS
            SELECT s1.entity_id AS source1_entity_id, tgt.entity_id AS matched_entity_id
            FROM test_s1 s1
            JOIN test_{target} tgt
              ON s1.country = tgt.country
             AND s1.house_norm = tgt.house_norm
             AND s1.name_prefix_2 = tgt.name_prefix_2
            WHERE s1.house_norm IS NOT NULL AND s1.house_norm <> ''
              AND LENGTH(s1.clean_name) >= 2 AND LENGTH(tgt.clean_name) >= 2
              AND jaro_winkler_similarity(s1.clean_name, tgt.clean_name) >= 0.80;
        """)
        
        # EXP-03 test
        con.execute(f"""
            CREATE OR REPLACE TABLE test_exp3_{target} AS
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
        """)
        
        # EXP-04 test
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
        
        out_test_path = EXP_DIR / f"test_candidate_pairs_{target}_cap70.tsv"
        con.execute(f"""
            CREATE OR REPLACE TABLE test_{target}_cap70 AS
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
        cnt = con.execute(f"SELECT count(*) FROM test_{target}_cap70").fetchone()[0]
        print(f"Exporting {cnt:,} candidate pairs to {out_test_path}...")
        con.execute(f"COPY test_{target}_cap70 TO '{out_test_path}' (HEADER, DELIMITER '\t');")
        print(f"Done in {time.time()-t0:.2f}s.")

    print("\n=== Step 3: Evaluating TRAIN Recall ===")
    con.execute("""
        CREATE OR REPLACE TABLE gt AS 
        SELECT 
            source1_entity_id,
            trim(unnest(string_split(matched_entity_ids, ','))) as target_entity_id,
            CASE WHEN target_entity_id LIKE 'S2-%' THEN 's2' ELSE 's3' END as target_source
        FROM read_csv_auto('data/train/train_ground_truth.tsv', delim='\t', header=True);
    """)
    gt_total = con.execute("SELECT count(*) FROM gt").fetchone()[0]
    gt_s2 = con.execute("SELECT count(*) FROM gt WHERE target_source = 's2'").fetchone()[0]
    gt_s3 = con.execute("SELECT count(*) FROM gt WHERE target_source = 's3'").fetchone()[0]

    s2_cands = con.execute("SELECT count(*) FROM train_s2_cap70").fetchone()[0]
    s3_cands = con.execute("SELECT count(*) FROM train_s3_cap70").fetchone()[0]
    total_train_cands = s2_cands + s3_cands

    s2_cap = con.execute("""
        SELECT count(*) FROM train_s2_cap70 c 
        JOIN gt ON c.source1_entity_id = gt.source1_entity_id AND c.matched_entity_id = gt.target_entity_id
        WHERE gt.target_source = 's2';
    """).fetchone()[0]

    s3_cap = con.execute("""
        SELECT count(*) FROM train_s3_cap70 c 
        JOIN gt ON c.source1_entity_id = gt.source1_entity_id AND c.matched_entity_id = gt.target_entity_id
        WHERE gt.target_source = 's3';
    """).fetchone()[0]

    total_cap = s2_cap + s3_cap
    total_missed = gt_total - total_cap
    s2_recall = s2_cap / gt_s2
    s3_recall = s3_cap / gt_s3
    combined_recall = total_cap / gt_total

    print(f"Train S2 Recall: {s2_recall:.6%} ({s2_cap:,} / {gt_s2:,})")
    print(f"Train S3 Recall: {s3_recall:.6%} ({s3_cap:,} / {gt_s3:,})")
    print(f"Train Combined Recall: {combined_recall:.6%} ({total_cap:,} / {gt_total:,})")
    print(f"Total Captured: {total_cap:,} | Total Missed: {total_missed:,}")

    print("\n=== Step 4: Evaluating TRAIN Combined Fanout ===")
    con.execute("CREATE OR REPLACE TABLE s1_train AS SELECT entity_id as s1_id FROM read_parquet('P1/data/entities/train/source1/train_s1_entities.parquet');")
    n_train_s1 = con.execute("SELECT count(*) FROM s1_train").fetchone()[0]
    
    con.execute("""
        CREATE OR REPLACE TABLE train_fanout AS
        SELECT s.s1_id, COALESCE(c.cnt, 0) as cnt
        FROM s1_train s
        LEFT JOIN (
            SELECT source1_entity_id, count(*) as cnt
            FROM (
                SELECT source1_entity_id FROM train_s2_cap70
                UNION ALL
                SELECT source1_entity_id FROM train_s3_cap70
            ) GROUP BY source1_entity_id
        ) c ON s.s1_id = c.source1_entity_id;
    """)
    tr_fan = con.execute("""
        SELECT 
            count(CASE WHEN cnt = 0 THEN 1 END),
            quantile_cont(cnt, 0.50),
            quantile_cont(cnt, 0.90),
            quantile_cont(cnt, 0.95),
            quantile_cont(cnt, 0.99),
            quantile_cont(cnt, 0.999),
            max(cnt)
        FROM train_fanout;
    """).fetchone()

    print(f"Train Zero-candidate S1: {tr_fan[0]:,} ({tr_fan[0]/n_train_s1:.4%})")
    print(f"Train p50:   {tr_fan[1]:.1f}")
    print(f"Train p90:   {tr_fan[2]:.1f}")
    print(f"Train p95:   {tr_fan[3]:.1f}")
    print(f"Train p99:   {tr_fan[4]:.1f}")
    print(f"Train p99.9: {tr_fan[5]:.1f}")
    print(f"Train Max:   {tr_fan[6]}")

    print("\n=== Step 5: Evaluating TEST Combined Fanout ===")
    n_test_s1 = con.execute("SELECT count(*) FROM test_s1").fetchone()[0]
    con.execute("""
        CREATE OR REPLACE TABLE test_fanout AS
        SELECT s.entity_id as s1_id, COALESCE(c.cnt, 0) as cnt
        FROM test_s1 s
        LEFT JOIN (
            SELECT source1_entity_id, count(*) as cnt
            FROM (
                SELECT source1_entity_id FROM test_s2_cap70
                UNION ALL
                SELECT source1_entity_id FROM test_s3_cap70
            ) GROUP BY source1_entity_id
        ) c ON s.entity_id = c.source1_entity_id;
    """)
    te_fan = con.execute("""
        SELECT 
            count(CASE WHEN cnt = 0 THEN 1 END),
            quantile_cont(cnt, 0.50),
            quantile_cont(cnt, 0.90),
            quantile_cont(cnt, 0.95),
            quantile_cont(cnt, 0.99),
            quantile_cont(cnt, 0.999),
            max(cnt)
        FROM test_fanout;
    """).fetchone()

    print(f"Test Zero-candidate S1: {te_fan[0]:,} ({te_fan[0]/n_test_s1:.4%})")
    print(f"Test p50:   {te_fan[1]:.1f}")
    print(f"Test p90:   {te_fan[2]:.1f}")
    print(f"Test p95:   {te_fan[3]:.1f}")
    print(f"Test p99:   {te_fan[4]:.1f}")
    print(f"Test p99.9: {te_fan[5]:.1f}")
    print(f"Test Max:   {te_fan[6]}")

    print("\n=== Step 6: Verifying Candidate Integrity Across All 4 Files ===")
    integrity_ok = True
    integrity_results = {}
    
    for split in ['train', 'test']:
        s1_pool = f"P1/data/entities/{split}/source1/{split}_s1_entities.parquet"
        con.execute(f"CREATE OR REPLACE TABLE s1_{split}_check AS SELECT entity_id, country_raw as country FROM read_parquet('{s1_pool}');")
        for target in ['s2', 's3']:
            tgt_pool = f"P1/data/entities/{split}/source{target[1]}/{split}_{target}_entities.parquet"
            con.execute(f"CREATE OR REPLACE TABLE tgt_{split}_{target}_check AS SELECT entity_id, country_raw as country FROM read_parquet('{tgt_pool}');")
            
            fpath = EXP_DIR / f"{split}_candidate_pairs_{target}_cap70.tsv"
            con.execute(f"CREATE OR REPLACE TABLE cand_chk AS SELECT source1_entity_id, matched_entity_id FROM read_csv_auto('{fpath}', delim='\t', header=True);")
            
            tot = con.execute("SELECT count(*) FROM cand_chk").fetchone()[0]
            uniq = con.execute("SELECT count(*) FROM (SELECT DISTINCT source1_entity_id, matched_entity_id FROM cand_chk)").fetchone()[0]
            dups = tot - uniq
            nulls = con.execute("SELECT count(*) FROM cand_chk WHERE source1_entity_id IS NULL OR matched_entity_id IS NULL").fetchone()[0]
            self_m = con.execute("SELECT count(*) FROM cand_chk WHERE source1_entity_id = matched_entity_id").fetchone()[0]
            inv_s1 = con.execute(f"SELECT count(*) FROM cand_chk c LEFT JOIN s1_{split}_check s ON c.source1_entity_id = s.entity_id WHERE s.entity_id IS NULL").fetchone()[0]
            inv_tgt = con.execute(f"SELECT count(*) FROM cand_chk c LEFT JOIN tgt_{split}_{target}_check t ON c.matched_entity_id = t.entity_id WHERE t.entity_id IS NULL").fetchone()[0]
            cross_c = con.execute(f"""
                SELECT count(*) FROM cand_chk c 
                JOIN s1_{split}_check s ON c.source1_entity_id = s.entity_id 
                JOIN tgt_{split}_{target}_check t ON c.matched_entity_id = t.entity_id 
                WHERE s.country <> t.country
            """).fetchone()[0]
            
            file_ok = (dups == 0 and nulls == 0 and self_m == 0 and inv_s1 == 0 and inv_tgt == 0 and cross_c == 0)
            if not file_ok:
                integrity_ok = False
            sha = get_file_sha256(fpath)
            size = os.path.getsize(fpath)
            
            integrity_results[f"{split}_{target}"] = {
                "path": str(fpath),
                "rows": tot,
                "bytes": size,
                "sha256": sha,
                "dups": dups,
                "nulls": nulls,
                "self_matches": self_m,
                "invalid_s1": inv_s1,
                "invalid_tgt": inv_tgt,
                "cross_country": cross_c,
                "pass": file_ok
            }
            print(f"{split.upper()} {target.upper()}: Tot={tot:,}, Dups={dups}, Nulls={nulls}, Self={self_m}, InvS1={inv_s1}, InvTgt={inv_tgt}, CrossCountry={cross_c} => {'PASS' if file_ok else 'FAIL'}")

    # Summary
    pass_recall = combined_recall > 0.657623
    pass_train_p95 = tr_fan[3] <= 300.0
    pass_test_p95 = te_fan[3] <= 300.0
    overall_pass = pass_recall and pass_train_p95 and pass_test_p95 and integrity_ok
    total_runtime = time.time() - t_global_start

    summary = {
        "experiment": "EXP-RET-01_CAP70",
        "description": "Controlled fanout correction experiment setting EXP-RET-01 per-S1 cap to 70",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_runtime_seconds": total_runtime,
        "v2_baseline_recall": 0.657623,
        "current_v3_metrics": {
            "train_recall": 0.724229,
            "train_p95": 199.0,
            "test_p95": 318.0,
            "train_zero_s1": 42613,
            "test_zero_s1": 33407
        },
        "cap70_metrics": {
            "train_candidate_count": total_train_cands,
            "train_s2_candidates": s2_cands,
            "train_s3_candidates": s3_cands,
            "test_s2_candidates": con.execute("SELECT count(*) FROM test_s2_cap70").fetchone()[0],
            "test_s3_candidates": con.execute("SELECT count(*) FROM test_s3_cap70").fetchone()[0],
            "gt_total": gt_total,
            "captured_true_pairs": total_cap,
            "missed_true_pairs": total_missed,
            "train_s2_recall": s2_recall,
            "train_s3_recall": s3_recall,
            "train_combined_recall": combined_recall,
            "train_combined_recall_delta_vs_v2": combined_recall - 0.657623,
            "train_combined_recall_delta_vs_v3": combined_recall - 0.724229,
            "train_fanout": {
                "zero_s1": tr_fan[0],
                "p50": tr_fan[1],
                "p90": tr_fan[2],
                "p95": tr_fan[3],
                "p99": tr_fan[4],
                "p999": tr_fan[5],
                "max": tr_fan[6]
            },
            "test_fanout": {
                "zero_s1": te_fan[0],
                "p50": te_fan[1],
                "p90": te_fan[2],
                "p95": te_fan[3],
                "p99": te_fan[4],
                "p999": te_fan[5],
                "max": te_fan[6]
            }
        },
        "integrity": integrity_results,
        "checks": {
            "train_recall_gt_657623": pass_recall,
            "train_p95_le_300": pass_train_p95,
            "test_p95_le_300": pass_test_p95,
            "candidate_integrity_zero_violations": integrity_ok,
            "overall_decision": "PASS" if overall_pass else "FAIL"
        }
    }

    with open(EXP_DIR / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved metrics to {EXP_DIR / 'metrics.json'}")
    print(f"Total Experiment Runtime: {total_runtime:.2f}s")
    print(f"OVERALL EXPERIMENT DECISION: {'PASS' if overall_pass else 'FAIL'}")

if __name__ == "__main__":
    main()
