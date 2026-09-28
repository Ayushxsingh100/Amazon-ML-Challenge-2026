#!/usr/bin/env python3
"""
f05_postprocess.py (v2)  --  tune the final match decision for Macro F0.5
Amazon ML Challenge 2026, Business Entity Resolution

Uses ONLY your own model scores + TRAIN labels (a held-out validation split).
It never uses test labels. Keep this file in your code package and describe
the chosen rule in your methodology write-up.

v2: finds the input files and the right columns by itself and prints what it
found; reads ids in any list format ("S2-1,S3-2", "['S2-1', 'S3-2']",
"S2-1|S3-2", or ids split into single characters like "S,2,-,1"); skips a
few unreadable rows with a warning instead of crashing; and stops with a clear
message when a file is incomplete (for example an upload that had not finished).

What it does
 1. Scores validation with the official metric: F0.5 per Source-1 entity,
    averaged over ALL S1 entities. A true singleton scores 1.0 only when
    nothing is predicted for it.
 2. Reports the CEILING of your current candidate list (perfect decisions on
    the candidates you already have).
 3. Tunes two families of decision rules on validation and keeps the better:
      A) per-source thresholds (S2, S3) + row gate + cut relative to the best
         candidate + max-matches cap
      B) expected-F0.5 choice of how many matches to keep per entity
    each optionally with "one S2/S3 record -> at most one S1" exclusivity.
 4. Applies the winner to your TEST candidate scores and writes the submission.

Inputs (file names or patterns like "validation_candidate_pairs_scored*"):
  VAL_SCORES   validation candidate pairs with the model score. The model must
               NOT have been trained on these S1 entities.
  VAL_TRUTH    train ground truth: S1 id + its matched S2/S3 ids.
  VAL_S1_LIST  optional: every validation S1 id (incl. ones with no match).
  TEST_SCORES  test candidate pairs with the model score.
  TEST_S1_LIST every test S1 id (your current matching_results.tsv works).
"""
import glob
import os
import re
import time

import numpy as np
import pandas as pd

CFG = dict(
    VAL_SCORES="validation_candidate_pairs_scor*",
    VAL_TRUTH="train_ground_truth*",
    VAL_S1_LIST="validation_s1_ids*",
    TEST_SCORES="test_candidate_pairs_scor*",
    TEST_S1_LIST="matching_results.tsv",
    OUTPUT="matching_results_tuned.tsv",
    VAL_CURRENT_PRED=None,            # optional: current validation predictions (submission format)
    SCORE_COLS=None,                  # None = auto-detect, or ("s1 column", "candidate column", "score column")
    CURRENT_THRESHOLD=None,           # e.g. 0.5 if you use one global cut-off today
    MAX_RANK=20,                      # consider only the top-N candidates per S1
    GRID_POINTS=60,                   # threshold grid size
    MAX_BAD_FRACTION=0.01,            # stop if more than 1% of a file's rows are unreadable
)

BASE = 10 ** 11          # "S2-123" -> 2 * BASE + 123
KEY = 10 ** 12           # (row, candidate) pair key = row * KEY + candidate
ID_FULL = r"S[1-3]-\d+"
CLEAN = r"[^A-Za-z0-9\-]"
EMPTY_WORDS = {"", "nan", "NaN", "None", "none", "null", "NULL"}
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


# ------------------------------------------------------------------ files
def resolve(pattern, what, required=True):
    if pattern is None:
        return None
    if os.path.isfile(pattern):
        return pattern
    hits = [h for h in glob.glob(pattern) if os.path.isfile(h)]
    if not hits:
        hits = [h for h in glob.glob(pattern + "*") if os.path.isfile(h)]
    if not hits:
        if required:
            raise SystemExit(f"{what}: no file matches {pattern!r}.\n"
                             f"Files in {os.getcwd()}: {sorted(os.listdir('.'))}")
        log(f"  {what}: {pattern!r} not found - continuing without it")
        return None
    hits.sort(key=lambda h: (not h.lower().endswith(".parquet"), len(h), h))
    if len(hits) > 1:
        log(f"  {what}: {len(hits)} files match {pattern!r}; using {hits[0]!r} (others: {hits[1:]})")
    return hits[0]


def kind_of(path):
    with open(path, "rb") as f:
        if f.read(4) == b"PAR1":
            return "parquet"
        f.seek(0)
        first = f.readline(1 << 20)
    return "tsv" if b"\t" in first else "csv"


def _sep(kind):
    return "\t" if kind == "tsv" else ","


def check_complete(path, kind, what):
    size = os.path.getsize(path)
    if size == 0:
        raise SystemExit(f"{what}: {path} is empty.")
    with open(path, "rb") as f:
        f.seek(max(0, size - 65536))
        tail = f.read()
        f.seek(0)
        header = f.readline(1 << 20).decode("utf-8", "replace").rstrip("\r\n")
    if kind == "parquet":
        if not tail.endswith(b"PAR1"):
            raise SystemExit(f"{what}: {path} is INCOMPLETE (parquet footer missing) - the upload had not "
                             "finished or was cut off. Wait for the upload to finish (or copy the file from "
                             "Google Drive), then run again.")
        return
    if tail.endswith(b"\n"):
        return
    last = tail.splitlines()[-1].decode("utf-8", "replace") if tail.strip() else ""
    if re.search(r"S(?:\d-?)?$", last) or len(last.split(_sep(kind))) < len(header.split(_sep(kind))):
        raise SystemExit(f"{what}: {path} looks CUT OFF - its last line is {last[-60:]!r} with no final "
                         "newline. The upload probably had not finished. Wait for it to finish, then run again.")
    log(f"  note: {path} has no final newline (last line {last[-60:]!r}) - fine if the upload had finished")


def _read_csv(path, kind, header, **kw):
    return pd.read_csv(path, sep=_sep(kind), dtype=str, keep_default_na=False,
                       header=0 if header else None, **kw)


def peek(path, kind, header=True, n=5000):
    if kind == "parquet":
        try:
            import pyarrow.parquet as pq
            return next(pq.ParquetFile(path).iter_batches(batch_size=n)).to_pandas()
        except ImportError:
            return pd.read_parquet(path).head(n)
    return _read_csv(path, kind, header, nrows=n)


def frames(t, columns=None, chunksize=2_000_000):
    path, kind, header = t["path"], t["kind"], t["header"]
    if kind == "parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError:
            yield pd.read_parquet(path, columns=columns)
            return
        for b in pq.ParquetFile(path).iter_batches(batch_size=chunksize, columns=columns):
            yield b.to_pandas()
    else:
        yield from _read_csv(path, kind, header, usecols=columns, chunksize=chunksize)


def open_table(pattern, what, required=True):
    path = resolve(pattern, what, required)
    if path is None:
        return None
    kind = kind_of(path)
    check_complete(path, kind, what)
    header = True
    sample = peek(path, kind)
    if kind != "parquet" and any(re.search(r"S[1-3]-\d", str(c)) for c in sample.columns):
        header = False                                     # first line is data, not a header
        sample = peek(path, kind, header=False)
    log(f"  {what}: {path}  ({os.path.getsize(path) / 1e6:,.1f} MB, {kind})  columns: {list(sample.columns)}")
    for rec in sample.head(2).astype(str).to_dict("records"):
        log(f"      e.g. {str(rec)[:170]}")
    return dict(path=path, kind=kind, header=header, sample=sample, what=what)


# ------------------------------------------------------------------- ids
def as_text(s):
    """any column -> strings; list/array cells are joined with commas"""
    nn = s.dropna()
    if len(nn) and isinstance(nn.iloc[0], (list, tuple, np.ndarray)):
        return s.map(lambda v: ",".join(map(str, v)) if isinstance(v, (list, tuple, np.ndarray)) else str(v))
    return s.astype(str)


def encode_one(s):
    """one id per cell -> int64 codes (-1 = unreadable), plus the raw text"""
    raw = as_text(s).str.strip()
    ids = raw.copy()
    full = raw.str.fullmatch(ID_FULL).fillna(False).to_numpy(dtype=bool)
    if not full.all():
        ids[~full] = raw[~full].str.replace(CLEAN, "", regex=True).str.extract(f"({ID_FULL})", expand=False)
    ok = ids.notna().to_numpy()
    out = np.full(len(raw), -1, dtype=np.int64)
    if ok.any():
        v = ids[ok].astype(str)
        if (v.str.len() > 14).any():
            raise SystemExit(f"ids longer than 11 digits are not supported, e.g. {v[v.str.len() > 14].iloc[0]!r}")
        if v.str.match(r"S\d-0\d").any():
            raise SystemExit(f"ids with leading zeros are not supported, e.g. {v[v.str.match(r'S[0-9]-0[0-9]')].iloc[0]!r}")
        out[ok] = (v.str.slice(1, 2).astype(np.int64).to_numpy() * BASE
                   + v.str.slice(3).astype(np.int64).to_numpy())
    return out, raw


def encode_many(s):
    """cells holding any number of S2/S3 ids -> (row position, code) pairs + mask of cells with junk"""
    raw = as_text(s)
    txt = raw.str.replace(CLEAN, "", regex=True)
    toks = txt.str.findall(r"S[23]-\d+")
    lens = toks.str.len().fillna(0).to_numpy(dtype=np.int64)
    rows = np.repeat(np.arange(len(raw)), lens)
    flat = toks.explode().dropna().reset_index(drop=True)
    codes = encode_one(flat)[0] if len(flat) else np.zeros(0, np.int64)
    left = txt.str.replace(ID_FULL, "", regex=True)
    junk = (~left.isin(EMPTY_WORDS)).to_numpy()
    return rows, codes, junk, raw


def _frac_ids(series, pattern):
    t = as_text(series).str.strip()
    t = t[~t.isin(EMPTY_WORDS)]
    return float(t.str.fullmatch(pattern).mean()) if len(t) else 0.0


def find_s1_col(t):
    sample = t["sample"]
    fr = {c: _frac_ids(sample[c], r"S1-\d+") for c in sample.columns}
    best = max(fr, key=fr.get) if fr else None
    if best is None or fr[best] < 0.8:
        raise SystemExit(f"{t['what']}: can't find a column of S1 ids (like 'S1-123') in {t['path']}.\n"
                         f"Columns: {list(sample.columns)}\nFirst rows:\n{sample.head(3).to_string()[:1500]}")
    return best


SCORE_NAME = re.compile(r"score|prob|pred|logit|conf|p_match|y_?hat", re.I)
LABEL_NAME = re.compile(r"label|target|is_?match|truth|gold|y_?true|^y$", re.I)
NOT_SCORE = re.compile(r"(^|_)(id|idx|index|rank|row|count|num|n)$|_id$", re.I)


def find_score_cols(t, override):
    sample, what = t["sample"], t["what"]
    cols = list(sample.columns)
    label_col = None
    if override:
        names = [cols[c] if isinstance(c, int) else c for c in override]
        missing = [n for n in names if n not in cols]
        if missing:
            raise SystemExit(f"{what}: SCORE_COLS {missing} not found. Columns are: {cols}")
        return list(names), None
    s1c = find_s1_col(t)
    rest = [c for c in cols if c != s1c]
    fr = {c: _frac_ids(sample[c], r"S[23]-\d+") for c in rest}
    cc = max(fr, key=fr.get) if fr else None
    if cc is None or fr[cc] < 0.8:
        raise SystemExit(f"{what}: can't find a column of S2/S3 candidate ids. Columns: {cols}\n"
                         f"First rows:\n{sample.head(3).to_string()[:1500]}")
    numeric, binary = [], []
    for c in rest:
        if c == cc:
            continue
        x = pd.to_numeric(as_text(sample[c]), errors="coerce")
        if x.notna().mean() < 0.99:
            continue
        if x.nunique() > 2:
            numeric.append(c)
        elif set(x.dropna().unique()) <= {0, 1}:
            binary.append(c)
    named = [c for c in numeric if SCORE_NAME.search(str(c)) and not LABEL_NAME.search(str(c))]
    pick = None
    for pref in ("score", "prob", "proba", "probability", "pred_prob", "match_prob", "pred_score",
                 "y_pred", "prediction", "pred"):
        pick = next((c for c in named if str(c).lower() == pref), None)
        if pick is not None:
            break
    if pick is None and named:
        pick = sorted(named, key=lambda c: ("score" not in str(c).lower(), "prob" not in str(c).lower(),
                                            len(str(c))))[0]
    if pick is None:
        plain = [c for c in numeric if not NOT_SCORE.search(str(c)) and not LABEL_NAME.search(str(c))]
        if len(plain) == 1:
            pick = plain[0]
    if pick is None:
        raise SystemExit(f"{what}: can't tell which column is the model score. Numeric columns: {numeric}.\n"
                         f"Set SCORE_COLS = ({s1c!r}, {cc!r}, '<score column>') in CFG and run again.")
    if len(named) > 1:
        log(f"  {what}: several score-like columns {named} -> using {pick!r} (set SCORE_COLS to change)")
    label_col = next((c for c in binary if LABEL_NAME.search(str(c))), None)
    log(f"  {what}: S1 column {s1c!r}, candidate column {cc!r}, score column {pick!r}"
        + (f", label column {label_col!r} (used only as a cross-check)" if label_col else ""))
    return [s1c, cc, pick], label_col


def _stop_if_bad(what, bad, n, examples, frac):
    if not bad:
        return
    log(f"  WARNING {what}: {bad:,} of {n:,} rows had unreadable ids and were skipped, e.g. {examples[:3]}")
    if bad > frac * n:
        raise SystemExit(f"{what}: too many unreadable rows ({bad / n:.1%}). The file format is not what the "
                         "script expects - paste the lines printed above (columns + examples) to get it fixed.")


def read_scores(pattern, what, c):
    t = open_table(pattern, what)
    cols, label_col = find_score_cols(t, c["SCORE_COLS"])
    use = cols + ([label_col] if label_col else [])
    S1, CA, SC, LB = [], [], [], []
    n = bad = 0
    ex = []
    for ch in frames(t, columns=use):
        a, _ = encode_one(ch[cols[0]])
        b, _ = encode_one(ch[cols[1]])
        col = ch[cols[2]]
        sc = (col.to_numpy(np.float64) if pd.api.types.is_numeric_dtype(col)
              else pd.to_numeric(as_text(col), errors="coerce").to_numpy(np.float64))
        ok = (a >= 0) & (b >= 0) & np.isfinite(sc)
        n += len(ch)
        if (~ok).any():
            bad += int((~ok).sum())
            if len(ex) < 3:
                ex += ch[cols][~ok].head(3).astype(str).values.tolist()
        S1.append(a[ok])
        CA.append(b[ok])
        SC.append(sc[ok])
        if label_col:
            LB.append(pd.to_numeric(as_text(ch[label_col]), errors="coerce").to_numpy(np.float64)[ok])
    _stop_if_bad(what, bad, n, ex, c["MAX_BAD_FRACTION"])
    log(f"  {what}: {n - bad:,} candidate pairs read")
    return (np.concatenate(S1), np.concatenate(CA), np.concatenate(SC),
            np.concatenate(LB) if label_col else None)


def read_lists(pattern, what, c, ids_only=False, required=True):
    """S1 id + matched ids (any list format). Returns (S1 ids in file order, pair S1, pair match, #rows)."""
    t = open_table(pattern, what, required)
    if t is None:
        return None
    sample = t["sample"]
    s1c = find_s1_col(t)
    id_cols = []
    if not ids_only:
        for col in sample.columns:
            if col == s1c:
                continue
            txt = as_text(sample[col]).str.replace(CLEAN, "", regex=True)
            txt = txt[~txt.isin(EMPTY_WORDS)]
            if len(txt) and txt.str.contains(r"S[23]-\d").mean() > 0.5:
                id_cols.append(col)
        if not id_cols:
            raise SystemExit(f"{what}: can't find a column with S2/S3 ids in {t['path']}. "
                             f"Columns: {list(sample.columns)}\nFirst rows:\n{sample.head(3).to_string()[:1500]}")
        for col in id_cols:
            raw = as_text(sample[col])
            parts = raw.str.split(",").explode().str.strip()
            parts = parts[parts != ""]
            if len(parts) and (parts.str.len() == 1).mean() > 0.5:
                eg = raw[raw.str.len() > 0].iloc[0][:40]
                log(f"  NOTE {what}: ids in column {col!r} are split into single characters (e.g. {eg!r}) "
                    "- joining them back. If your own validation code read this file, its scores were wrong too.")
        log(f"  {what}: S1 column {s1c!r}, matched-id column(s) {id_cols}")
    order, ps1, pm = [], [], []
    n = bad = junk_n = 0
    ex, junk_ex = [], []
    for ch in frames(t, columns=[s1c] + id_cols):
        s1v, raw1 = encode_one(ch[s1c])
        n += len(ch)
        b = (s1v < 0) & ~raw1.isin(EMPTY_WORDS).to_numpy()
        if b.any():
            bad += int(b.sum())
            ex += raw1[b].head(3).tolist()
        order.append(s1v[s1v >= 0])
        for col in id_cols:
            rows, codes, junk, raw = encode_many(ch[col])
            keep = s1v[rows] >= 0
            ps1.append(s1v[rows][keep])
            pm.append(codes[keep])
            if junk.any():
                junk_n += int(junk.sum())
                junk_ex += raw[junk].head(3).str.slice(0, 60).tolist()
    _stop_if_bad(what, bad, n, ex, c["MAX_BAD_FRACTION"])
    if junk_n:
        log(f"  WARNING {what}: {junk_n:,} cells contained text that is not a valid id "
            f"(ignored), e.g. {junk_ex[:3]}")
        if junk_n > c["MAX_BAD_FRACTION"] * n:
            raise SystemExit(f"{what}: too many cells with unreadable ids ({junk_n / n:.1%}). "
                             "Paste the lines printed above to get the format fixed.")
    ps1 = np.concatenate(ps1) if ps1 else np.zeros(0, np.int64)
    pm = np.concatenate(pm) if pm else np.zeros(0, np.int64)
    return pd.unique(np.concatenate(order)), ps1, pm, n


def decode(v):
    v = pd.Series(np.asarray(v, dtype=np.int64))
    return ("S" + (v // BASE).astype(str) + "-" + (v % BASE).astype(str)).to_numpy()


def index_of(universe, values):
    order = np.argsort(universe, kind="stable")
    su = universe[order]
    pos = np.clip(np.searchsorted(su, values), 0, max(len(su) - 1, 0))
    return np.where(su[pos] == values, order[pos], -1) if len(su) else np.full(len(values), -1)


# ------------------------------------------------------------ candidate arrays
class Cands:
    pass


def _groups(keys):
    starts = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]]) if len(keys) else np.zeros(0, np.int64)
    return starts, np.diff(np.r_[starts, len(keys)])


def build(universe, s1, cand, score, max_rank, truth=None):
    A = Cands()
    A.N = len(universe)
    row = index_of(universe, s1)
    keep = row >= 0
    if (~keep).any():
        log(f"  ignored {int((~keep).sum()):,} candidate rows whose S1 id is not in the S1 list")
    row, cand, score = row[keep], cand[keep], score[keep]
    o = np.lexsort((-score, cand, row))                     # one row per (S1, candidate)
    row, cand, score = row[o], cand[o], score[o]
    dup = np.r_[False, (row[1:] == row[:-1]) & (cand[1:] == cand[:-1])]
    row, cand, score = row[~dup], cand[~dup], score[~dup]
    o = np.lexsort((-score, row))                           # by S1, best score first
    row, cand, score = row[o], cand[o], score[o]
    starts, counts = _groups(row)
    rank = np.arange(len(row)) - np.repeat(starts, counts)
    k = rank < max_rank
    row, cand, score, rank = row[k], cand[k], score[k], rank[k]
    A.starts, A.counts = _groups(row)
    A.row, A.cand, A.score, A.rank = row.astype(np.int64), cand, score, rank
    A.grp = np.repeat(np.arange(len(A.starts)), A.counts)
    A.smax = np.repeat(score[A.starts], A.counts)
    A.is_s2 = (cand // BASE) == 2
    A.order2 = np.lexsort((-score, cand))                   # for exclusivity
    A.starts2, A.counts2 = _groups(cand[A.order2])
    if truth is not None:
        ts1, tm = truth
        trow = index_of(universe, ts1)
        tk = trow >= 0
        A.key_t = np.unique(trow[tk].astype(np.int64) * KEY + tm[tk])
        A.nt = np.bincount(A.key_t // KEY, minlength=A.N)
        A.label = np.isin(A.row * KEY + A.cand, A.key_t)
    return A


# ------------------------------------------------------------------- metric
def f05_rows(nt, npred, tp):
    denom = 0.25 * nt + npred
    f = np.divide(1.25 * tp, denom, out=np.zeros(len(nt)), where=denom > 0)
    return np.where(nt == 0, (npred == 0).astype(float), f)


def macro(A, sel):
    npred = np.bincount(A.row, weights=sel, minlength=A.N)
    tp = np.bincount(A.row, weights=sel & A.label, minlength=A.N)
    return f05_rows(A.nt, npred, tp).mean()


# ------------------------------------------------------------ decision rules
def _cap(A, sel, k):
    cs = np.cumsum(sel, dtype=np.int64)
    before = np.repeat(cs[A.starts] - sel[A.starts], A.counts)
    return sel & ((cs - before) <= k)


def _exclusive(A, sel):
    s = sel[A.order2]
    cs = np.cumsum(s, dtype=np.int64)
    before = np.repeat(cs[A.starts2] - s[A.starts2], A.counts2)
    out = np.zeros_like(sel)
    out[A.order2] = s & ((cs - before) == 1)
    return out


def _post(A, sel, p):
    if p.get("k", 0) > 0:
        sel = _cap(A, sel, p["k"])
    if p.get("excl"):
        sel = _exclusive(A, sel)
    return sel


def rule_A(A, p):
    ok = A.score >= np.where(A.is_s2, p["t2"], p["t3"])
    if p["r"] > 0:
        ok &= A.score >= p["r"] * A.smax
    sel = (A.smax >= p["t_row"]) & ((A.rank == 0) | ok)
    return _post(A, sel, p)


def fit_calibration(score, label, bins=400):
    qs = np.unique(np.quantile(score, np.linspace(0, 1, bins + 1)))
    if len(qs) < 2:
        rate = float(label.mean())
        return lambda s: np.full(len(s), rate)
    idx = np.clip(np.searchsorted(qs, score, side="right") - 1, 0, len(qs) - 2)
    cnt = np.bincount(idx, minlength=len(qs) - 1).astype(float)
    pos = np.bincount(idx, weights=label, minlength=len(qs) - 1)
    ctr = np.bincount(idx, weights=score, minlength=len(qs) - 1)
    m = cnt > 0
    cnt, pos, ctr = cnt[m], pos[m], ctr[m] / cnt[m]
    blocks = []                                             # pool-adjacent-violators
    for c, w, x in zip(pos, cnt, ctr):
        blocks.append([c, w, [x]])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            b = blocks.pop()
            blocks[-1][0] += b[0]
            blocks[-1][1] += b[1]
            blocks[-1][2] += b[2]
    xs = np.array([x for b in blocks for x in b[2]])
    ys = np.array([b[0] / b[1] for b in blocks for _ in b[2]])
    return lambda s: np.interp(s, xs, ys)


def prep_B(A, calib):
    p = np.clip(calib(A.score), 1e-6, 1 - 1e-6)
    ends = A.starts + A.counts - 1
    cs = np.cumsum(p)
    A.Sk = cs - np.repeat(cs[A.starts] - p[A.starts], A.counts)
    A.Sp = np.repeat(A.Sk[ends], A.counts)
    lg = np.cumsum(np.log1p(-p))
    within = lg - np.repeat(lg[A.starts] - np.log1p(-p[A.starts]), A.counts)
    A.logP0 = np.repeat(within[ends], A.counts)


def rule_B(A, p):
    P0 = np.exp(A.logP0 - p["m"])                           # P(entity has no match)
    nt_if_any = (A.Sp + p["m"]) / np.maximum(1.0 - P0, 1e-9)
    E = 1.25 * A.Sk / (0.25 * nt_if_any + A.rank + 1)       # expected F0.5 keeping top-(rank+1)
    rowmax = np.repeat(np.maximum.reduceat(E, A.starts), A.counts)
    is_max = E >= rowmax
    cs = np.cumsum(is_max, dtype=np.int64)
    first = is_max & ((cs - np.repeat(cs[A.starts] - is_max[A.starts], A.counts)) == 1)
    kstar = np.zeros(len(A.starts), dtype=np.int64)
    kstar[A.grp[first]] = A.rank[first] + 1
    sel = (rowmax > p["lam"] * P0) & (A.rank < kstar[A.grp])
    return _post(A, sel, p)


# ------------------------------------------------------------------- search
def tune(A, fn, p, space, rounds=3):
    cur = macro(A, fn(A, p))
    for rnd in range(rounds):
        improved = False
        for name, values_fn in space:
            best_v, best_s = p[name], cur
            for v in values_fn(p, rnd):
                q = dict(p)
                q[name] = v
                s = macro(A, fn(A, q))
                if s > best_s + 1e-7:
                    best_v, best_s = v, s
            if best_s > cur + 1e-6:
                log(f"    {name}: {fmt({0: p[name]})[0]} -> {fmt({0: best_v})[0]}   validation {cur:.5f} -> {best_s:.5f}")
                p[name], cur, improved = best_v, best_s, True
        if not improved:
            break
    return p, cur


def thr_values(grid, name):
    def fn(p, rnd):
        if rnd == 0:
            return grid
        v = p[name]
        i = int(np.clip(np.searchsorted(grid, v), 1, len(grid) - 1))
        lo, hi = grid[max(i - 2, 0)], grid[min(i + 1, len(grid) - 1)]
        return np.unique(np.r_[grid, np.linspace(lo, hi, 25)])
    return fn


def fmt(p):
    return {k: (round(float(v), 5) if isinstance(v, (float, np.floating)) else v) for k, v in p.items()}


def shape(A, sel):
    npred = np.bincount(A.row, weights=sel, minlength=A.N)
    return (npred == 0).mean(), npred.mean()


# --------------------------------------------------------------------- main
def main(c=CFG):
    log("Reading validation files ...")
    _, ts1, tm, _ = read_lists(c["VAL_TRUTH"], "VAL_TRUTH", c)
    vs1, vcand, vscore, vlab = read_scores(c["VAL_SCORES"], "VAL_SCORES", c)
    lst = read_lists(c["VAL_S1_LIST"], "VAL_S1_LIST", c, ids_only=True, required=False) if c["VAL_S1_LIST"] else None
    if lst is not None:
        uni = lst[0]
    else:
        uni = pd.unique(vs1)
        log("  note: no VAL_S1_LIST - using the S1 ids found in VAL_SCORES (entities with zero candidates"
            " are not counted, so scores are a close approximation)")
    truth_s1 = np.unique(ts1)
    cover = np.isin(uni, truth_s1).mean()
    log(f"  validation S1 entities: {len(uni):,}  ({cover:.1%} of them have at least one true match)")
    if len(uni) > 0.6 * max(len(np.unique(np.r_[truth_s1, uni])), 1) and len(uni) > 100_000:
        log("  CHECK: validation looks as large as the whole training set - make sure these scores come from a"
            " model that did NOT train on these entities (out-of-fold), or the tuned rule will be too optimistic.")

    A = build(uni, vs1, vcand, vscore, c["MAX_RANK"], truth=(ts1, tm))
    if A.label.sum() == 0:
        raise SystemExit("No candidate matches a true pair - VAL_TRUTH and VAL_SCORES don't share ids. "
                         "Check that VAL_TRUTH is the train ground truth and VAL_SCORES are train/validation pairs.")
    if vlab is not None:
        r = index_of(uni, vs1)
        k = (r >= 0) & np.isfinite(vlab)
        t_lab = np.isin(r[k].astype(np.int64) * KEY + vcand[k], A.key_t)
        f_lab = vlab[k] >= 0.5
        agree = (t_lab == f_lab).mean()
        log(f"  cross-check: truth file vs label column in VAL_SCORES agree on {agree:.2%} of pairs")
        if agree < 0.97:
            log("  WARNING: they disagree a lot - one of the two files is wrong, so results below may be off.")
    log(f"  mean score: true matches {A.score[A.label].mean():.4f} vs non-matches {A.score[~A.label].mean():.4f}")
    flip = A.score[A.label].mean() < A.score[~A.label].mean()
    if flip:
        log("  NOTE: true matches have LOWER scores than non-matches - treating the score as a distance (sign flipped)")
        A = build(uni, vs1, vcand, -vscore, c["MAX_RANK"], truth=(ts1, tm))
    del vs1, vcand, vscore, vlab

    n_true_links = len(A.key_t)
    found = int(A.label.sum())
    tp_rows = np.bincount(A.row, weights=A.label, minlength=A.N)
    ceiling = f05_rows(A.nt, tp_rows, tp_rows).mean()
    lost = int(((A.nt > 0) & (tp_rows == 0)).sum())
    print()
    log(f"Validation entities: {A.N:,}   singletons (no true match): {(A.nt == 0).mean():.2%}"
        f"   mean true matches: {A.nt.mean():.3f}")
    log(f"Candidate pairs used: {len(A.row):,}   candidate recall: {found / max(n_true_links, 1):.2%}"
        f"   entities whose matches never reach the candidate list: {lost:,} ({lost / A.N:.2%})")
    log(f"CEILING with your current candidates (perfect decisions): {ceiling:.5f}")
    if ceiling < 0.99:
        log("  -> 0.99 is out of reach by post-processing alone; the rest has to come from better blocking/candidates.")

    if c["VAL_CURRENT_PRED"]:
        cur = read_lists(c["VAL_CURRENT_PRED"], "VAL_CURRENT_PRED", c, required=False)
        if cur is not None:
            r = index_of(uni, cur[1])
            key = np.unique(r[r >= 0].astype(np.int64) * KEY + cur[2][r >= 0])
            npred = np.bincount(key // KEY, minlength=A.N)
            tp = np.bincount(key[np.isin(key, A.key_t)] // KEY, minlength=A.N)
            log(f"Your current validation predictions score: {f05_rows(A.nt, npred, tp).mean():.5f}")

    top = A.score[A.rank < 3]
    grid = np.unique(np.round(np.quantile(top, np.linspace(0.02, 0.995, c["GRID_POINTS"])), 6))
    base = dict(t_row=grid[0], t2=grid[0], t3=grid[0], r=0.0, k=0, excl=False)
    if c["CURRENT_THRESHOLD"] is not None and not flip:
        t = float(c["CURRENT_THRESHOLD"])
        s = macro(A, rule_A(A, dict(base, t_row=t, t2=t, t3=t)))
        log(f"Single cut-off {t} (your current rule, approx.): {s:.5f}")
    best_t, best_s = grid[0], -1.0
    for t in grid:
        s = macro(A, rule_A(A, dict(base, t_row=t, t2=t, t3=t)))
        if s > best_s:
            best_t, best_s = t, s
    log(f"Best single cut-off {best_t:.4f}: {best_s:.5f}")

    print()
    log("Tuning rule A (per-source thresholds, row gate, relative cut, cap, exclusivity) ...")
    pA = dict(base, t_row=best_t, t2=best_t, t3=best_t)
    r_vals = [0.0] if A.score.min() < 0 else [0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    gate = np.r_[-np.inf, grid]
    spaceA = [("t2", thr_values(grid, "t2")), ("t3", thr_values(grid, "t3")),
              ("t_row", lambda p, rnd: gate if rnd == 0 else thr_values(grid, "t_row")(p, rnd)),
              ("r", lambda p, rnd: r_vals), ("k", lambda p, rnd: [0, 1, 2, 3, 4, 5, 6, 8, 10]),
              ("excl", lambda p, rnd: [False, True])]
    pA, sA = tune(A, rule_A, pA, spaceA)
    log(f"Rule A: {sA:.5f}")

    print()
    log("Tuning rule B (expected-F0.5 per entity on calibrated probabilities) ...")
    calib = fit_calibration(A.score, A.label.astype(float))
    prep_B(A, calib)
    pB = dict(lam=1.0, m=0.0, k=0, excl=False)
    spaceB = [("lam", lambda p, rnd: [0.3, 0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0, 3.0] if rnd == 0
               else np.round(np.linspace(p["lam"] * 0.8, p["lam"] * 1.25, 19), 4)),
              ("m", lambda p, rnd: [0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]),
              ("k", lambda p, rnd: [0, 1, 2, 3, 4, 5, 6, 8, 10]),
              ("excl", lambda p, rnd: [False, True])]
    pB, sB = tune(A, rule_B, pB, spaceB)
    log(f"Rule B: {sB:.5f}")

    use_B = sB > sA
    fn, p, s = (rule_B, pB, sB) if use_B else (rule_A, pA, sA)
    e_val, m_val = shape(A, fn(A, p))
    print()
    log(f"WINNER: rule {'B' if use_B else 'A'}  validation macro F0.5 = {s:.5f}"
        f"  (best single cut-off was {best_s:.5f}, ceiling {ceiling:.5f})")
    log(f"  parameters: {fmt(p)}")
    log(f"  validation shape - predicted empty: {e_val:.2%} vs true singletons {(A.nt == 0).mean():.2%};"
        f" predicted matches/entity {m_val:.3f} vs true {A.nt.mean():.3f}")
    del A

    print()
    log("Reading test files ...")
    tuni, os1, om, _ = read_lists(c["TEST_S1_LIST"], "TEST_S1_LIST", c)
    s1, cand, sc, _ = read_scores(c["TEST_SCORES"], "TEST_SCORES", c)
    T = build(tuni, s1, cand, -sc if flip else sc, c["MAX_RANK"])
    del s1, cand, sc
    if use_B:
        prep_B(T, calib)
    sel = fn(T, p)
    rows, cands = T.row[sel], T.cand[sel]
    out = np.full(T.N, "", dtype=object)
    if len(rows):
        strs = decode(cands)
        b, n = _groups(rows)
        out[rows[b]] = [",".join(strs[i:i + j]) for i, j in zip(b, n)]
    sub = pd.DataFrame({"source1_entity_id": decode(tuni), "matched_entity_ids": out})
    assert sub["source1_entity_id"].is_unique and len(sub) == len(tuni)
    sub.to_csv(c["OUTPUT"], sep="\t", index=False)
    e_t = (out == "").mean()
    log(f"Wrote {c['OUTPUT']}: {len(sub):,} rows, empty {e_t:.2%}, matches/entity {len(rows) / max(T.N, 1):.3f}")
    r = index_of(tuni, os1)
    old = np.bincount(r[r >= 0], minlength=len(tuni))
    log(f"  your current test file: empty {(old == 0).mean():.2%}, matches/entity {old.mean():.3f}")
    if abs(e_t - e_val) > 0.05:
        log("  CHECK: test empty-rate differs from validation by >5 points - possible train/test shift.")


if __name__ == "__main__":
    main()
