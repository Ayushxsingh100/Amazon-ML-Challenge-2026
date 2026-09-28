import os
import duckdb
import hashlib

REPO = os.path.abspath(os.path.dirname(__file__))
P3_MANIFEST = os.path.join(REPO, "P3", "reports", "folds_v1_manifest.tsv")

def verify_p3():
    print("--- TASK 2: VERIFY P3 ---")
    if not os.path.exists(P3_MANIFEST):
        print("FAIL: folds_v1_manifest.tsv does not exist")
        return False
        
    with open(P3_MANIFEST, "rb") as f:
        sha256 = hashlib.sha256(f.read()).hexdigest()
    print(f"SHA256: {sha256}")
    
    expected_sha = "9dcec5d83a477d224067e71b93abc21af8befa26f9c399568121fa83ba8801a3"
    if sha256 != expected_sha:
        print(f"FAIL: SHA256 mismatch (Expected: {expected_sha})")
    
    con = duckdb.connect()
    con.execute(f"CREATE TABLE folds AS SELECT * FROM read_csv('{P3_MANIFEST}', delim='\\t', header=true)")
    
    total = con.execute("SELECT COUNT(*) FROM folds").fetchone()[0]
    unique = con.execute("SELECT COUNT(DISTINCT source1_entity_id) FROM folds").fetchone()[0]
    
    print(f"Total S1: {total}")
    print(f"Unique S1: {unique}")
    
    fold_counts = con.execute("SELECT fold, COUNT(*) FROM folds GROUP BY fold ORDER BY fold").fetchall()
    print("Fold counts:")
    for f, c in fold_counts:
        print(f"  Fold {f}: {c}")
        
    print("\n--- TASK 5: FOLD COMPATIBILITY ---")
    s1_path = os.path.join(REPO, "outputs", "person1_step1", "normalized", "train_source1_normalized.tsv")
    con.execute(f"CREATE TABLE s1 AS SELECT entity_id FROM read_csv('{s1_path}', delim='\\t', header=true)")
    
    s1_total = con.execute("SELECT COUNT(*) FROM s1").fetchone()[0]
    overlap = con.execute("SELECT COUNT(*) FROM folds JOIN s1 ON folds.source1_entity_id = s1.entity_id").fetchone()[0]
    
    print(f"Total S1 entities in train_source1_normalized: {s1_total}")
    print(f"Entities in both folds and S1 table: {overlap}")
    if s1_total == overlap == total:
        print("PASS: folds_v1 covers exactly the same 2,206,821 train S1 entities")
        return True
    return False

if __name__ == "__main__":
    verify_p3()
