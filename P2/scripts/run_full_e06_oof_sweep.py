#!/usr/bin/env python3
"""
Full E06 OOF Threshold Sweep for P3 Validation
==============================================
- Authoritative OOF: P2/reports/E06_V2_OOF_FULL_CANONICAL.tsv (67,332,524 candidate rows)
- Authoritative Ground Truth: outputs/person1_step1/train_ground_truth_reconstructed.tsv (2,206,821 S1 entities)
- Scorer: Exact canonical Macro F0.5 as defined in validation/scorer_v1.py & validation/metrics.py
- Single streaming pass over all 67,332,524 rows evaluating thresholds:
    [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
- Outputs:
    P2/reports/E06_FULL_OOF_THRESHOLD_SWEEP.tsv
    P2/reports/E06_FULL_OOF_THRESHOLD_SWEEP.md
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import duckdb

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OOF_PATH = os.path.join(REPO, "P2", "reports", "E06_V2_OOF_FULL_CANONICAL.tsv")
GT_PATH = os.path.join(REPO, "outputs", "person1_step1", "train_ground_truth_reconstructed.tsv")
OUT_TSV = os.path.join(REPO, "P2", "reports", "E06_FULL_OOF_THRESHOLD_SWEEP.tsv")
OUT_MD = os.path.join(REPO, "P2", "reports", "E06_FULL_OOF_THRESHOLD_SWEEP.md")

THRESHOLDS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]


def main():
    t_start = time.time()
    print("=" * 70, flush=True)
    print("FULL E06 OOF THRESHOLD SWEEP (P3 VALIDATION ONLY)", flush=True)
    print("=" * 70, flush=True)

    # 1. Load Ground Truth & build entity index mapping
    print(f"Loading Ground Truth from {GT_PATH} ...", flush=True)
    t0 = time.time()
    gt_df = pd.read_csv(GT_PATH, sep="\t", dtype=str)
    n_s1 = len(gt_df)
    print(f"Loaded {n_s1:,} train S1 entities in {time.time()-t0:.2f}s", flush=True)
    assert n_s1 == 2206821, f"Expected 2,206,821 S1 entities, got {n_s1:,}"

    s1_ids = gt_df["source1_entity_id"].values
    s1_to_idx = {sid: i for i, sid in enumerate(s1_ids)}

    # Compute actual ground-truth lengths per entity
    actual_lens = np.zeros(n_s1, dtype=np.int32)
    for i, m_ids in enumerate(gt_df["matched_entity_ids"]):
        if pd.notna(m_ids) and str(m_ids).strip():
            actual_lens[i] = len(str(m_ids).strip().split(","))

    no_match_gt_count = int((actual_lens == 0).sum())
    matched_gt_count = n_s1 - no_match_gt_count
    print(f"Ground Truth breakdown: {matched_gt_count:,} matched entities, {no_match_gt_count:,} no-match entities", flush=True)

    del gt_df

    # 2. Setup accumulators for the 9 thresholds
    num_thresh = len(THRESHOLDS)
    thresh_arr = np.array(THRESHOLDS, dtype=np.float32)

    # pred_counts[t_idx, s1_idx] and tp_counts[t_idx, s1_idx]
    pred_counts = np.zeros((num_thresh, n_s1), dtype=np.int32)
    tp_counts = np.zeros((num_thresh, n_s1), dtype=np.int32)

    # 3. Stream over OOF file
    print(f"\nStreaming over full OOF file: {OOF_PATH} ...", flush=True)
    total_rows = 0
    nan_count = 0
    inf_count = 0

    CHUNK_SIZE = 5000000
    chunk_idx = 0

    # Use pd.read_csv with chunksize for exact, deterministic parsing
    for chunk in pd.read_csv(
        OOF_PATH,
        sep="\t",
        chunksize=CHUNK_SIZE,
        dtype={
            "source1_entity_id": str,
            "matched_entity_id": str,
            "target": np.int8,
            "fold": np.int8,
            "oof_score": np.float32,
        },
    ):
        t_chunk = time.time()
        chunk_len = len(chunk)
        total_rows += chunk_len
        chunk_idx += 1

        scores = chunk["oof_score"].values
        targets = chunk["target"].values
        s1_str = chunk["source1_entity_id"].values

        # Integrity checks: no NaNs / Infs
        nan_count += int(np.isnan(scores).sum())
        inf_count += int(np.isinf(scores).sum())
        assert nan_count == 0, f"Found NaN in oof_score at chunk {chunk_idx}"
        assert inf_count == 0, f"Found Inf in oof_score at chunk {chunk_idx}"

        # Map s1 IDs to indices
        s1_indices = np.fromiter((s1_to_idx[s] for s in s1_str), dtype=np.int32, count=chunk_len)

        is_pos = (targets == 1)

        # Accumulate for each threshold
        for t_i, t_val in enumerate(THRESHOLDS):
            mask = scores >= t_val
            if np.any(mask):
                np.add.at(pred_counts[t_i], s1_indices[mask], 1)
                pos_mask = mask & is_pos
                if np.any(pos_mask):
                    np.add.at(tp_counts[t_i], s1_indices[pos_mask], 1)

        print(
            f"  Chunk {chunk_idx} ({chunk_len:,} rows) | "
            f"Processed: {total_rows:,} / 67,332,524 ({total_rows/67332524*100:.1f}%) | "
            f"Time: {time.time()-t_chunk:.1f}s",
            flush=True,
        )

    print(f"\nCompleted OOF stream: {total_rows:,} candidate rows processed.", flush=True)
    assert total_rows == 67332524, f"Expected 67,332,524 rows, got {total_rows:,}"

    # 4. Compute exact Macro F0.5 metrics for all 9 thresholds
    print("\n" + "=" * 70, flush=True)
    print("COMPUTING EXACT SCORER_V1 MACRO F0.5 METRICS", flush=True)
    print("=" * 70, flush=True)

    results = []

    for t_i, t_val in enumerate(THRESHOLDS):
        p_cnt = pred_counts[t_i]
        tp_cnt = tp_counts[t_i]
        act_cnt = actual_lens

        fp_cnt = p_cnt - tp_cnt
        fn_cnt = act_cnt - tp_cnt

        # Entity-level calculation matching validation/metrics.py
        # Case 1: act_cnt == 0 and p_cnt == 0 -> precision=1, recall=1, f05=1
        # Case 2: act_cnt == 0 and p_cnt > 0  -> precision=0, recall=0, f05=0
        # Case 3: act_cnt > 0:
        #   precision = tp / p_cnt if p_cnt > 0 else 0.0
        #   recall    = tp / act_cnt
        #   f05       = (1.25 * P * R) / (0.25 * P + R) if (0.25 * P + R) > 0 else 0.0

        f05_arr = np.zeros(n_s1, dtype=np.float64)

        # Case 1: correct no-match
        correct_no_match_mask = (act_cnt == 0) & (p_cnt == 0)
        f05_arr[correct_no_match_mask] = 1.0
        correct_no_match = int(correct_no_match_mask.sum())

        # Case 2: false positive for no-match entity -> remains 0.0

        # Case 3: act_cnt > 0
        has_gt_mask = act_cnt > 0
        p_has_gt = p_cnt[has_gt_mask]
        tp_has_gt = tp_cnt[has_gt_mask]
        act_has_gt = act_cnt[has_gt_mask]

        prec = np.where(p_has_gt > 0, tp_has_gt / p_has_gt, 0.0)
        rec = tp_has_gt / act_has_gt
        denom = 0.25 * prec + rec
        f05_has_gt = np.where(denom > 0, (1.25 * prec * rec) / denom, 0.0)

        f05_arr[has_gt_mask] = f05_has_gt

        macro_f05 = float(np.mean(f05_arr))
        total_tp = int(np.sum(tp_cnt))
        total_fp = int(np.sum(fp_cnt))
        total_fn = int(np.sum(fn_cnt))
        non_empty_s1 = int(np.sum(p_cnt > 0))

        row_dict = {
            "threshold": f"{t_val:.2f}",
            "macro_f0_5": f"{macro_f05:.9f}",
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "correct_no_match_count": correct_no_match,
            "non_empty_predicted_s1": non_empty_s1,
            "candidate_rows": total_rows,
            "s1_entities": n_s1,
        }
        results.append(row_dict)

        print(
            f"  T = {t_val:.2f} | "
            f"Macro F0.5 = {macro_f05:.9f} | "
            f"TP = {total_tp:,} | "
            f"FP = {total_fp:,} | "
            f"FN = {total_fn:,} | "
            f"Correct No-Match = {correct_no_match:,} | "
            f"Non-Empty S1 = {non_empty_s1:,}",
            flush=True,
        )

    # 5. Task 6 Verification at T=0.50
    t050_row = [r for r in results if r["threshold"] == "0.50"][0]
    t050_score = float(t050_row["macro_f0_5"])
    expected_score = 0.747085258
    diff = abs(t050_score - expected_score)
    print(f"\nT=0.50 Verification: computed = {t050_score:.9f}, expected = {expected_score:.9f} (diff = {diff:.2e})", flush=True)
    assert diff < 1e-6, f"Discrepancy at T=0.50! Computed {t050_score}, expected {expected_score}"
    print("  -> REPRODUCED AUTHORITATIVE P3 T=0.50 SCORE EXACTLY!", flush=True)

    # 6. Save results to TSV
    res_df = pd.DataFrame(results)
    res_df.to_csv(OUT_TSV, sep="\t", index=False)
    print(f"\nSaved TSV report: {OUT_TSV}", flush=True)

    # 7. Save Markdown Report
    best_row = max(results, key=lambda r: float(r["macro_f0_5"]))
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(f"""# E06 Full OOF Threshold Sweep Report

**Experiment**: E06 (`cands_BCD_v2` 67,332,524 full-canonical OOF)  
**Date**: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}  
**Status**: COMPLETE (Full population evaluated without sampling)  
**Total Candidate Rows**: 67,332,524  
**Total S1 Entities**: 2,206,821  
**Scorer**: Canonical `validation/scorer_v1.py` (`macro_f0.5`)  

---

## 1. Full Population Threshold Sweep Results

| Threshold | Macro F0.5 | True Positives (TP) | False Positives (FP) | False Negatives (FN) | Correct No-Match | Non-Empty S1 | Candidate Rows | S1 Entities |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
""")
        for r in results:
            f.write(f"| **{r['threshold']}** | **{r['macro_f0_5']}** | {r['tp']:,} | {r['fp']:,} | {r['fn']:,} | {r['correct_no_match_count']:,} | {r['non_empty_predicted_s1']:,} | {r['candidate_rows']:,} | {r['s1_entities']:,} |\n")

        f.write(f"""
---

## 2. Key Findings & Verification

1. **Authoritative T=0.50 Match**: Computed **`{t050_score:.9f}`** perfectly reproduces the authoritative P3 value (`0.747085258`).
2. **Best OOF Threshold**: **$T = {best_row['threshold']}$** achieving Macro F0.5 = **`{best_row['macro_f0_5']}`** (+{(float(best_row['macro_f0_5'])-t050_score):.6f} over $T=0.50$).
3. **FP Elimination Curve**: Moving from $T=0.50$ to $T=0.90$ reduces False Positives from {int(t050_row['fp']):,} down to {int([r for r in results if r['threshold'] == '0.90'][0]['fp']):,} (a {(1 - int([r for r in results if r['threshold'] == '0.90'][0]['fp'])/int(t050_row['fp']))*100:.1f}% reduction).
4. **Zero Missingness / Full Integrity**: Zero NaNs or Infs across all 67,332,524 candidate rows.

---

## 3. Policy & Governance Notice
This sweep was conducted for validation and diagnostic analysis only. No changes have been applied to test submission artifacts. Any decision regarding threshold adjustments for future submissions is governed by Person 3.
""")

    print(f"Saved Markdown report: {OUT_MD}", flush=True)

    total_time = time.time() - t_start
    print(f"\nALL TASKS COMPLETED IN {total_time:.1f}s ({total_time/60:.2f} min)", flush=True)


if __name__ == "__main__":
    main()
