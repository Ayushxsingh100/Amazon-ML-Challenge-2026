#!/usr/bin/env python3
"""
Build Final Submission Files:
1. output/candidate_pairs.tsv
2. output/matching_results.tsv
"""

import os
import sys
import time
from collections import defaultdict
import duckdb

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
NORM_DIR = os.path.join(REPO, "outputs", "person1_step1", "normalized")
CAND_DIR = os.path.join(REPO, "P2", "data", "candidates")
TMP_DIR = os.path.join(REPO, "P2", "data", "duckdb_tmp_test_inference")
OUTPUT_DIR = os.path.join(REPO, "output")

os.makedirs(OUTPUT_DIR, exist_ok=True)

TEST_S1_PATH = os.path.join(NORM_DIR, "test_source1_normalized.tsv")
TEST_S2_CAND = os.path.join(CAND_DIR, "test_candidate_pairs_s2_v2.tsv")
TEST_S3_CAND = os.path.join(CAND_DIR, "test_candidate_pairs_s3_v2.tsv")

TMP_MATCHES_S2 = os.path.join(TMP_DIR, "tmp_matches_s2.tsv")
TMP_MATCHES_S3 = os.path.join(TMP_DIR, "tmp_matches_s3.tsv")

OUTPUT_CANDIDATES = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
OUTPUT_MATCHING = os.path.join(OUTPUT_DIR, "matching_results.tsv")


def build_candidate_pairs():
    print("=" * 70)
    print("TASK 3: BUILDING output/candidate_pairs.tsv")
    print("=" * 70, flush=True)
    t0 = time.time()

    # 1. Read all canonical test S1 entities
    print("Reading canonical test S1 entities...", flush=True)
    s1_ids = []
    with open(TEST_S1_PATH, "r", encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        id_idx = header.index("entity_id")
        for line in f:
            if not line.strip():
                continue
            s1_ids.append(line.split("\t")[id_idx])

    s1_ids.sort()
    n_s1 = len(s1_ids)
    print(f"Total S1 entities: {n_s1:,}", flush=True)

    # 2. Build candidate dict using defaultdict(list)
    print("Reading S2 candidates into memory...", flush=True)
    cands_map = defaultdict(list)
    total_pairs = 0

    with open(TEST_S2_CAND, "r", encoding="utf-8") as f:
        f.readline()  # skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                cands_map[parts[0]].append(parts[1])
                total_pairs += 1

    print(f"S2 candidates loaded. Total pairs so far: {total_pairs:,}", flush=True)

    print("Reading S3 candidates into memory...", flush=True)
    with open(TEST_S3_CAND, "r", encoding="utf-8") as f:
        f.readline()  # skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                cands_map[parts[0]].append(parts[1])
                total_pairs += 1

    print(f"Total candidate pairs loaded: {total_pairs:,}", flush=True)
    assert total_pairs == 76633796, f"Expected 76,633,796 candidate pairs, got {total_pairs:,}"

    # 3. Write output/candidate_pairs.tsv
    print(f"Writing {OUTPUT_CANDIDATES} ...", flush=True)
    with open(OUTPUT_CANDIDATES, "w", encoding="utf-8") as out_f:
        out_f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in s1_ids:
            if s1_id in cands_map:
                cands = cands_map[s1_id]
                cands.sort()
                out_f.write(f"{s1_id}\t{','.join(cands)}\n")
            else:
                out_f.write(f"{s1_id}\t\n")

    size_mb = os.path.getsize(OUTPUT_CANDIDATES) / (1024 * 1024)
    print(f"Wrote {OUTPUT_CANDIDATES} ({size_mb:.1f} MB) in {time.time()-t0:.1f}s", flush=True)


def build_matching_results():
    print("\n" + "=" * 70)
    print("TASK 4: BUILDING output/matching_results.tsv")
    print("=" * 70, flush=True)
    t0 = time.time()

    # 1. Read all canonical test S1 entities
    s1_ids = []
    with open(TEST_S1_PATH, "r", encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        id_idx = header.index("entity_id")
        for line in f:
            if not line.strip():
                continue
            s1_ids.append(line.split("\t")[id_idx])

    s1_ids.sort()
    n_s1 = len(s1_ids)

    # 2. Build match dict
    matches_map = defaultdict(list)
    total_matches = 0

    for path in [TMP_MATCHES_S2, TMP_MATCHES_S3]:
        print(f"Reading matches from {path} ...", flush=True)
        with open(path, "r", encoding="utf-8") as f:
            f.readline()  # skip header
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    matches_map[parts[0]].append(parts[1])
                    total_matches += 1

    print(f"Total predicted matches loaded: {total_matches:,}", flush=True)
    assert total_matches == 9928563, f"Expected 9,928,563 matches, got {total_matches:,}"

    # 3. Write output/matching_results.tsv
    print(f"Writing {OUTPUT_MATCHING} ...", flush=True)
    with open(OUTPUT_MATCHING, "w", encoding="utf-8") as out_f:
        out_f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in s1_ids:
            if s1_id in matches_map:
                m_list = matches_map[s1_id]
                m_list.sort()
                out_f.write(f"{s1_id}\t{','.join(m_list)}\n")
            else:
                out_f.write(f"{s1_id}\t\n")

    size_mb = os.path.getsize(OUTPUT_MATCHING) / (1024 * 1024)
    print(f"Wrote {OUTPUT_MATCHING} ({size_mb:.1f} MB) in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    build_candidate_pairs()
    build_matching_results()
