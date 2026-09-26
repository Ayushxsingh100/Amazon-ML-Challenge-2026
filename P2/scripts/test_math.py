import sys
import os

REPO = r"c:\NEW AMAZON\Amazon-ML-Challenge-2026"
sys.path.insert(0, REPO)
from validation.metrics import entity_precision_recall_f05

# Test various cases
cases = [
    (set(), set()),
    ({'B'}, set()),
    (set(), {'A'}),
    ({'A'}, {'A'}),
    ({'A', 'B'}, {'A'}),
    ({'A'}, {'A', 'B'}),
    ({'A', 'B'}, {'A', 'C'}),
    ({'A', 'B', 'C'}, {'A', 'B', 'C', 'D'}),
]

for pred, act in cases:
    res = entity_precision_recall_f05(pred, act)
    A = len(act)
    P = len(pred)
    TP = len(pred & act)
    if A == 0:
        calc_f05 = 1.0 if P == 0 else 0.0
    else:
        calc_f05 = (5.0 * TP) / (A + 4.0 * P) if TP > 0 else 0.0
    diff = abs(res['f0.5'] - calc_f05)
    print(f"Pred: {pred}, Act: {act} -> Metric: {res['f0.5']:.6f}, Calc: {calc_f05:.6f}, Diff: {diff}")
    assert diff < 1e-12
print("All test cases matched exactly!")
