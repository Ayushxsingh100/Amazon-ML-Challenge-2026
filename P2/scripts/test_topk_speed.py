import time
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

print("--- Benchmarking Sparse TF-IDF Top-K ---")
t0 = time.time()

# Mock 100,000 docs with ~20,000 vocab
vocab_size = 20000
num_docs = 200000
nnz_per_doc = 60

# Generate random sparse matrix representing TF-IDF
data = np.random.rand(num_docs * nnz_per_doc).astype(np.float32)
rows = np.repeat(np.arange(num_docs), nnz_per_doc)
cols = np.random.randint(0, vocab_size, size=num_docs * nnz_per_doc)
D = sp.csr_matrix((data, (rows, cols)), shape=(num_docs, vocab_size))

# Normalize rows to unit length (L2)
norm = sp.linalg.norm(D, axis=1)
norm[norm == 0] = 1.0
D = sp.diags(1.0 / norm.ravel()) @ D

print(f"Built mock doc matrix ({num_docs} x {vocab_size}) in {time.time() - t0:.2f}s, size: {D.data.nbytes / 1e6:.1f} MB")

# Test query chunk of 2,000 queries
num_queries = 2000
q_data = np.random.rand(num_queries * nnz_per_doc).astype(np.float32)
q_rows = np.repeat(np.arange(num_queries), nnz_per_doc)
q_cols = np.random.randint(0, vocab_size, size=num_queries * nnz_per_doc)
Q = sp.csr_matrix((q_data, (q_rows, q_cols)), shape=(num_queries, vocab_size))
q_norm = sp.linalg.norm(Q, axis=1)
q_norm[q_norm == 0] = 1.0
Q = sp.diags(1.0 / q_norm.ravel()) @ Q

t_mult = time.time()
# Sparse matrix multiplication: Q (2000 x 20000) @ D.T (20000 x 200000)
# But note: Q @ D.T may produce a very dense result if every query shares at least one common 3-gram with almost every doc!
# Let's check how dense the result is!
sims = Q @ D.T
print(f"Multiplication shape: {sims.shape}, nnz: {sims.nnz}, time: {time.time() - t_mult:.2f}s")
