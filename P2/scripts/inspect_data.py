import duckdb
import os

REPO = r"c:\NEW AMAZON\Amazon-ML-Challenge-2026"
FOLDS_PATH = os.path.join(REPO, "P3", "reports", "folds_v1_manifest.tsv")
GT_PATH = os.path.join(REPO, "outputs", "person1_step1", "train_ground_truth_reconstructed.tsv")
OOF_PATH = os.path.join(REPO, "P2", "reports", "E02_OOF_FULL_CANONICAL.tsv")

con = duckdb.connect()
print("Checking folds...")
con.execute(f"CREATE TABLE folds AS SELECT source1_entity_id, fold FROM read_csv('{FOLDS_PATH}', delim='\\t', header=true)")
print("Folds count:", con.execute("SELECT COUNT(*), COUNT(DISTINCT source1_entity_id) FROM folds").fetchall())

print("Checking GT...")
con.execute(f"CREATE TABLE gt AS SELECT source1_entity_id, matched_entity_ids FROM read_csv('{GT_PATH}', delim='\\t', header=true)")
print("GT count:", con.execute("SELECT COUNT(*), COUNT(DISTINCT source1_entity_id) FROM gt").fetchall())
print("GT sample:", con.execute("SELECT * FROM gt LIMIT 5").fetchall())
print("GT nulls:", con.execute("SELECT COUNT(*) FROM gt WHERE matched_entity_ids IS NULL OR matched_entity_ids = ''").fetchall())

print("Checking OOF...")
con.execute(f"CREATE TABLE oof_sample AS SELECT * FROM read_csv('{OOF_PATH}', delim='\\t', header=true) LIMIT 10")
print("OOF sample:", con.execute("SELECT * FROM oof_sample").df())
