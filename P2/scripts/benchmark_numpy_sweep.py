import time
import numpy as np

# Benchmark numpy bincount and score calculation
N = 440272
M = 1045434

s1_idx = np.random.randint(0, N, size=M, dtype=np.int32)
target = np.random.randint(0, 2, size=M, dtype=np.int8)
oof_score = np.random.uniform(0.50, 1.0, size=M).astype(np.float32)
A = np.random.randint(0, 5, size=N, dtype=np.int16)

thresholds = np.arange(0.500, 0.996, 0.005, dtype=np.float32)
print(f"Number of thresholds: {len(thresholds)}")

t0 = time.time()
scores = []
for t in thresholds:
    mask = oof_score >= t
    p_idx = s1_idx[mask]
    p_tgt = target[mask]
    
    P = np.bincount(p_idx, minlength=N).astype(np.float32)
    TP = np.bincount(p_idx, weights=p_tgt, minlength=N).astype(np.float32)
    
    f05 = np.where(
        A == 0,
        np.where(P == 0, 1.0, 0.0),
        np.where(TP > 0, (5.0 * TP) / (A + 4.0 * P), 0.0)
    )
    scores.append(np.sum(f05))

elapsed = time.time() - t0
print(f"Evaluated {len(thresholds)} thresholds in {elapsed:.4f}s ({elapsed/len(thresholds)*1000:.2f} ms/threshold)")
