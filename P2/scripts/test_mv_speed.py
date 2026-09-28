import time
import numpy as np
import scipy.sparse as sp

# Let's test multiplying 100 queries against Dt
# where Dt is (V x N)
# If we do D @ q for a single query vector q (V x 1):
# D is (N x V) sparse, q is (V x 1) sparse or dense
# D @ q produces a (N x 1) dense vector!
# Size of (N x 1) float32 for 730,000 docs is ONLY 2.9 MB!
# And argpartition top 5 on 730,000 floats takes ~0.002s in numpy!
num_docs = 730000
vocab_size = 15000
nnz = 40

d_data = np.random.rand(num_docs * nnz).astype(np.float32)
d_rows = np.repeat(np.arange(num_docs), nnz)
d_cols = np.random.randint(0, vocab_size, size=num_docs * nnz)
D = sp.csr_matrix((d_data, (d_rows, d_cols)), shape=(num_docs, vocab_size))

# Single query vector q
q = np.random.rand(vocab_size).astype(np.float32)

t0 = time.time()
for _ in range(100):
    sims = D @ q  # 1D array of length 730,000
    top5 = np.argpartition(-sims, 5)[:5]
dt = time.time() - t0
print(f"Time for 100 queries: {dt:.3f}s ({dt/100*1000:.2f} ms/query)")
print(f"Rate: {100 / dt:.1f} queries/sec")
