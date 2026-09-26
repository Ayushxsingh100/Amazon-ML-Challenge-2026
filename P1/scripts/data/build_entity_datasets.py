#!/usr/bin/env python3
"""
Phase 1 — Canonical Entity Dataset Builder
Amazon ML Challenge 2026 — Person 1 (P1)

Builds standardized, lossless, deterministic Parquet entity datasets for:
- train S1, S2, S3
- test S1, S2, S3

Preserves 100% of raw fields (business_name_raw, business_address_raw, country_raw)
and pairs them with normalized text and deterministic derived blocking keys.
All outputs are strictly sorted by entity_id ASC to ensure determinism.
"""

import os
import sys
import time
import hashlib
import duckdb

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

DATA_CONFIG = [
    {
        "split": "train",
        "source": "source1",
        "raw_path": os.path.join(REPO_ROOT, "data", "train", "train_source1.tsv"),
        "norm_path": os.path.join(REPO_ROOT, "outputs", "person1_step1", "normalized", "train_source1_normalized.tsv"),
        "out_dir": os.path.join(REPO_ROOT, "P1", "data", "entities", "train", "source1"),
        "out_filename": "train_s1_entities.parquet"
    },
    {
        "split": "train",
        "source": "source2",
        "raw_path": os.path.join(REPO_ROOT, "data", "train", "train_source2.tsv"),
        "norm_path": os.path.join(REPO_ROOT, "outputs", "person1_step1", "normalized", "train_source2_normalized.tsv"),
        "out_dir": os.path.join(REPO_ROOT, "P1", "data", "entities", "train", "source2"),
        "out_filename": "train_s2_entities.parquet"
    },
    {
        "split": "train",
        "source": "source3",
        "raw_path": os.path.join(REPO_ROOT, "data", "train", "train_source3.tsv"),
        "norm_path": os.path.join(REPO_ROOT, "outputs", "person1_step1", "normalized", "train_source3_normalized.tsv"),
        "out_dir": os.path.join(REPO_ROOT, "P1", "data", "entities", "train", "source3"),
        "out_filename": "train_s3_entities.parquet"
    },
    {
        "split": "test",
        "source": "source1",
        "raw_path": os.path.join(REPO_ROOT, "data", "test", "test_source1.tsv"),
        "norm_path": os.path.join(REPO_ROOT, "outputs", "person1_step1", "normalized", "test_source1_normalized.tsv"),
        "out_dir": os.path.join(REPO_ROOT, "P1", "data", "entities", "test", "source1"),
        "out_filename": "test_s1_entities.parquet"
    },
    {
        "split": "test",
        "source": "source2",
        "raw_path": os.path.join(REPO_ROOT, "data", "test", "test_source2.tsv"),
        "norm_path": os.path.join(REPO_ROOT, "outputs", "person1_step1", "normalized", "test_source2_normalized.tsv"),
        "out_dir": os.path.join(REPO_ROOT, "P1", "data", "entities", "test", "source2"),
        "out_filename": "test_s2_entities.parquet"
    },
    {
        "split": "test",
        "source": "source3",
        "raw_path": os.path.join(REPO_ROOT, "data", "test", "test_source3.tsv"),
        "norm_path": os.path.join(REPO_ROOT, "outputs", "person1_step1", "normalized", "test_source3_normalized.tsv"),
        "out_dir": os.path.join(REPO_ROOT, "P1", "data", "entities", "test", "source3"),
        "out_filename": "test_s3_entities.parquet"
    }
]

def build_entities():
    con = duckdb.connect()
    con.execute("PRAGMA threads=4;")
    con.execute("PRAGMA memory_limit='8GB';")

    common_prefixes = "('the', 'shri', 'sri', 'dr', 'm/s', 'hotel', 'new', 'om', 'sai', 'jai', 'a', 'an')"

    for cfg in DATA_CONFIG:
        os.makedirs(cfg["out_dir"], exist_ok=True)
        out_path = os.path.join(cfg["out_dir"], cfg["out_filename"])
        print(f"\n========================================================")
        print(f"Building Canonical Entity Dataset: {cfg['split']} {cfg['source']}")
        print(f"  Raw Input: {cfg['raw_path']}")
        print(f"  Norm Input: {cfg['norm_path']}")
        print(f"  Destination: {out_path}")
        print(f"========================================================")

        t0 = time.time()
        query = f"""
        COPY (
            SELECT 
                raw.entity_id,
                raw.business_name AS business_name_raw,
                raw.business_address AS business_address_raw,
                raw.country AS country_raw,
                norm.business_name AS business_name_normalized,
                norm.business_address AS business_address_normalized,
                norm.country AS country_normalized,
                regexp_replace(lower(trim(coalesce(norm.business_name, ''))), '[^a-z0-9]', '', 'g') AS name_clean,
                regexp_replace(lower(trim(coalesce(norm.business_address, ''))), '[^a-z0-9]', '', 'g') AS address_clean,
                string_split(trim(regexp_replace(coalesce(norm.business_name, ''), '\\\s+', ' ', 'g')), ' ') AS name_tokens,
                string_split(trim(regexp_replace(coalesce(norm.business_address, ''), '\\\s+', ' ', 'g')), ' ') AS address_tokens,
                LEFT(TRIM(coalesce(norm.business_name, '')), 1) AS name_prefix_1,
                LEFT(TRIM(coalesce(norm.business_name, '')), 2) AS name_prefix_2,
                LEFT(TRIM(coalesce(norm.business_name, '')), 3) AS name_prefix_3,
                LEFT(TRIM(coalesce(norm.business_name, '')), 4) AS name_prefix_4,
                split_part(TRIM(coalesce(norm.business_name, '')), ' ', 1) AS first_token,
                CASE 
                    WHEN lower(split_part(TRIM(coalesce(norm.business_name, '')), ' ', 1)) IN {common_prefixes} 
                         AND split_part(TRIM(coalesce(norm.business_name, '')), ' ', 2) <> '' 
                    THEN split_part(TRIM(coalesce(norm.business_name, '')), ' ', 2)
                    ELSE split_part(TRIM(coalesce(norm.business_name, '')), ' ', 1)
                END AS root_token,
                regexp_extract(TRIM(coalesce(norm.business_address, '')), '[0-9]+[A-Za-z]?', 0) AS house_number,
                regexp_replace(regexp_extract(TRIM(coalesce(norm.business_address, '')), '[0-9]+[A-Za-z]?', 0), '^0+', '') AS house_number_norm,
                regexp_extract(TRIM(coalesce(norm.business_address, '')), '(?:^|[^0-9])([0-9]{{5,6}})(?:[^0-9]|$)', 1) AS postal_code,
                regexp_replace(lower(trim(coalesce(norm.business_name, ''))), '[^a-z0-9]', '', 'g') AS name_alnum,
                regexp_replace(lower(trim(coalesce(norm.business_address, ''))), '[^a-z0-9]', '', 'g') AS address_alnum
            FROM read_csv('{cfg["raw_path"]}', sep='\\t', header=true, all_varchar=true) raw
            JOIN read_csv('{cfg["norm_path"]}', sep='\\t', header=true, all_varchar=true) norm
              ON raw.entity_id = norm.entity_id
            ORDER BY raw.entity_id ASC
        ) TO '{out_path}' (FORMAT PARQUET, COMPRESSION ZSTD);
        """
        con.execute(query)
        dt = time.time() - t0
        sz_bytes = os.path.getsize(out_path)
        sz_mb = sz_bytes / (1024 * 1024)

        # Quick row count check
        r_cnt = con.execute(f"SELECT COUNT(*) FROM read_parquet('{out_path}')").fetchone()[0]
        print(f"Generated {out_path}: {r_cnt:,} rows, {sz_mb:.2f} MB in {dt:.2f}s")

if __name__ == "__main__":
    build_entities()
