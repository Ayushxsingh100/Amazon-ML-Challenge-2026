import os
import duckdb

con = duckdb.connect()

print("--- Checking columns of normalized TSVs ---")
for src in ["train_source1_normalized.tsv", "train_source2_normalized.tsv", "train_source3_normalized.tsv"]:
    p = f"Amazon-ML-Challenge-2026/outputs/person1_step1/normalized/{src}"
    schema = con.execute(f"DESCRIBE SELECT * FROM read_csv('{p}', delim='\\t', header=True, all_varchar=True)").fetchall()
    print(src, ":", [col[0] for col in schema])

print("\n--- Checking raw parts in data/train ---")
for src in ["train_source1.tsv.part_aa", "train_source2.tsv.part_aa", "train_source3.tsv.part_aa"]:
    p = f"Amazon-ML-Challenge-2026/data/train/{src}"
    schema = con.execute(f"DESCRIBE SELECT * FROM read_csv('{p}', delim='\\t', header=True, all_varchar=True)").fetchall()
    print(src, ":", [col[0] for col in schema])

print("\n--- Checking P1 parquet entities ---")
for src in ["P1/data/entities/train/source2/train_s2_entities.parquet", "P1/data/entities/train/source3/train_s3_entities.parquet"]:
    p = f"Amazon-ML-Challenge-2026/{src}"
    schema = con.execute(f"DESCRIBE SELECT * FROM '{p}'").fetchall()
    print(src, ":", [col[0] for col in schema])
