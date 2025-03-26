import pandas as pd
import numpy as np
import pickle
import platform
from datetime import datetime
import faiss
import importlib.metadata
import insightface

from benchmark_helper import (
    faiss_search_flat,
    faiss_search_ivf,
    faiss_search_hnsw,
    faiss_search_innerproduct, 
    faiss_search_ivfpq,
    faiss_search_ivfpqr,
    hnswlib_search, 
    brute_force_search
)

# load data
filepath = "/mnt/nas2/sabrina/face-gen/embeddings_sabrina.pkl"
df = pd.read_pickle(filepath)
is_array_col = df['embedding'].apply(lambda x: isinstance(x, np.ndarray))

# convert back to float32
df['embedding'] = df['embedding'].apply(lambda x: x.astype(np.float32))

# set params
top_k = 10
test_mode = False

# IVF / PQ/R params
nlist = 100
m = 32
nbits = 8
nprobe = 10

# hnswlib params
efc = 200
efs = 200

results = []

# collect info
machine_name = platform.node()
timestamp = datetime.now().isoformat()
faiss_version = getattr(faiss, '__version__', 'unknown')

try:
    hnswlib_version = importlib.metadata.version('hnswlib')
except importlib.metadata.PackageNotFoundError:
    hnswlib_version = 'unknown'

# ====== Run All Index Types ======

# FAISS Flat
print("\nRunning FAISS Flat Benchmark...")
flat_result = faiss_search_flat(df, top_k=10, test_mode=test_mode)
flat_result.update({
    "index_type": "flat",
    "machine": machine_name,
    "timestamp": timestamp, 
    "faiss_version": faiss_version
})
results.append(flat_result)

# FAISS IVF
print("\nRunning FAISS IVF Benchmark...")
ivf_result = faiss_search_ivf(df, top_k=10, nlist=100, test_mode=test_mode)
ivf_result.update({
    "index_type": "ivf",
    "nlist": 100,
    "machine": machine_name,
    "timestamp": timestamp, 
    "faiss_version": faiss_version
})
results.append(ivf_result)

# FAISS HNSW
print("\nRunning FAISS HNSW Benchmark...")
hnsw_result = faiss_search_hnsw(df, top_k=10, m=32, test_mode=test_mode)
hnsw_result.update({
    "index_type": "hnsw",
    "M": 32,
    "machine": machine_name,
    "timestamp": timestamp,
    "faiss_version": faiss_version
})
results.append(hnsw_result)

# FAISS INNER PRODUCT
print("\nRunning FAISS Inner Product Benchmark...")
innerprod_result = faiss_search_innerproduct(df, top_k=top_k, normalize=False, test_mode=test_mode)
innerprod_result.update({
    "index_type": "innerproduct",
    "normalize": False,
    "machine": machine_name,
    "timestamp": timestamp,
    "faiss_version": faiss_version
})
results.append(innerprod_result)

# FAISS IVFPQ
print("\nRunning FAISS IVFPQ Benchmark...")
ivfpq_result = faiss_search_ivfpq(df, nlist=nlist, m=m, nbits=nbits, test_mode=test_mode)
ivfpq_result.update({
    "index_type": "ivfpq",
    "nlist": 100,
    "M": 16,
    "nbits": 8,
    "machine": machine_name,
    "timestamp": timestamp,
    "faiss_version": faiss_version
})
results.append(ivfpq_result)

# FAISS IVFPQR
"""print("\nRunning FAISS IVFPQR Benchmark...")
ivfpqr_result = faiss_search_ivfpqr(df, nlist=nlist, m=m, nbits=nbits, nprobe=nprobe, test_mode=test_mode)
ivfpqr_result.update({
    "index_type": "ivfpqr",
    "nlist": 100,
    "M": 16,
    "nbits": 8,
    'nprobe': 10,
    "machine": machine_name,
    "timestamp": timestamp,
    "faiss_version": faiss_version
})
results.append(ivfpq_result)"""


# hnswlib
print("\nRunning hnswlib Benchmark...")
hnswlib_result = hnswlib_search(df, efc=efc, m=m, efs=efs, top_k=top_k, test_mode=test_mode)
hnswlib_result.update({
    "index_type": "hnswlib",
    "M": 32,
    "machine": machine_name,
    "timestamp": timestamp,
    "hnswlib_version": hnswlib_version
})
results.append(hnswlib_result)


# brute force
print("\nRunning brute force Benchmark...")
detector = insightface.model_zoo.get_model('/home/sho/.insightface/models/buffalo_l/w600k_r50.onnx')

brute_force_result = brute_force_search(df, detector, top_k=top_k, test_mode=test_mode)
brute_force_result.update({
    "index_type": "brute_force",
    "machine": machine_name,
    "timestamp": timestamp,
})
results.append(brute_force_result)



df_results = pd.DataFrame(results)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
df_results.to_csv(f"cpu_benchmark_results_{timestamp}.csv", index=False)
print("CSV saved!")