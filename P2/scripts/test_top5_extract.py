import time
import numpy as np
import scipy.sparse as sp

# Test chunk of 1000 queries against 500,000 docs
num_docs = 500000
vocab_size = 20000
nnz = 50

# Mock D
d_data = np.random.rand(num_docs * nnz).astype(np.float32)
d_rows = np.repeat(np.arange(num_docs), nnz)
d_cols = np.random.randint(0, vocab_size, size=num_docs * nnz)
D = sp.csr_matrix((d_data, (d_rows, d_cols)), shape=(num_docs, vocab_size))
# Unit norm
norm = sp.linalg.norm(D, axis=1)
norm[norm == 0] = 1.0
D = sp.diags(1.0 / norm.ravel()) @ D
# Transpose for dot product
Dt = D.T.tocsc()

# Mock Q
num_q = 1000
q_data = np.random.rand(num_q * nnz).astype(np.float32)
q_rows = np.repeat(np.arange(num_q), nnz)
q_cols = np.random.randint(0, vocab_size, size=num_q * nnz)
Q = sp.csr_matrix((q_data, (q_rows, q_cols)), shape=(num_q, vocab_size))
norm_q = sp.linalg.norm(Q, axis=1)
norm_q[norm_q == 0] = 1.0
Q = sp.diags(1.0 / norm_q.ravel()) @ Q

t0 = time.time()
# Dense dot product of chunk: (1000 x 20000) @ (20000 x 500000)
# Instead of full sparse matmul which creates giant sparse matrix:
# For each query or smaller batch:
# Notice Q (csr) @ Dt (csc)
res = Q @ Dt  # csr matrix (1000, 500000)
print(f"Matmul time: {time.time() - t0:.2f}s, nnz={res.nnz}")

t1 = time.time()
# Extract top 5 for each row of csr matrix
top5_per_row = []
for i in range(num_q):
    start = res.indptr[i]
    end = res.indptr[i+1]
    row_data = res.data[start:end]
    row_indices = res.indices[start:end]
    if len(row_data) <= 5:
        top_idx = row_indices[np.argsort(-row_data)]
    else:
        # argpartition top 5
        part = np.argpartition(-row_data, 5)[:5]
        top_idx = row_indices[part[np.argsort(-row_data[part])]]
    top5_per_row.append(top_idx)

print(f"Top 5 extraction time: {time.time() - t1:.2f}s")
