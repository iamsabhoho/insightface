import numpy as np
import pandas as pd
import faiss
import platform
from datetime import datetime
from tqdm import tqdm
import hnswlib

##### HELPERS #####
def evaluate_topk(query_id, top_k_ids, k):
    return query_id in top_k_ids[:k]

def log_topk_failure(query_idx, query_id, query_name, top_k_ids, k):
    return {
        'query_idx': query_idx,
        'query_id': query_id,
        'query_name': query_name,
        'top_k_ids': top_k_ids[:k]
    }

##### SEARCH #####
def _evaluate_faiss_index(df, embedding_matrix, index_algo, top_k=10, test_mode=False):
    dim = embedding_matrix.shape[1]
    total = 0
    top_k_metrics = {1: 0, 3: 0, 10: 0}
    top_k_failures = {1: [], 3: [], 10: []}
    search_time = []

    if test_mode:
        print("##### test mode #####")

    for query_idx in tqdm(range(len(df)), desc="FAISS Exclude-Self Search"):
        if test_mode and query_idx >= 100:
            break

        query_id = df.iloc[query_idx]['ID']
        query_name = df.iloc[query_idx]['name']
        query_vector = embedding_matrix[query_idx].reshape(1, -1)

        if df[df.ID == query_id].shape[0] == 1:
            continue

        total += 1

        # Remove query from dataset
        mask = np.ones(len(df), dtype=bool)
        mask[query_idx] = False
        reduced_embeddings = embedding_matrix[mask]
        reduced_df = df[mask].reset_index(drop=True)

        # Build and search
        index = index_algo(dim, reduced_embeddings)
        start_time = datetime.now()
        _, indices = index.search(query_vector, top_k)
        end_time = datetime.now()

        search_time.append((end_time - start_time).total_seconds()) 

        top_k_indices = indices[0]
        top_k_ids = reduced_df.iloc[top_k_indices]['ID'].tolist()

        # Evaluate each top-k level
        for k in [1, 3, 10]:
            if evaluate_topk(query_id, top_k_ids, k):
                top_k_metrics[k] += 1
            else:
                top_k_failures[k].append(log_topk_failure(query_idx, query_id, query_name, top_k_ids, k))

    # Final reporting
    metrics = {f'top{k}_acc': top_k_metrics[k] / total for k in [1, 3, 10]}
    metrics.update({f'top{k}_failures': top_k_failures[k] for k in [1, 3, 10]})
    metrics['median_latency'] = np.median(search_time) * 1000

    print(f"Total queries evaluated: {total}")
    for k in [1, 3, 10]:
        print(f"Top-{k} Accuracy: {metrics[f'top{k}_acc']:.4f}")
        print(f"Top-{k} Failures: {len(metrics[f'top{k}_failures'])}")
    print(f"Median Query Time: {metrics['median_latency']:.4f} ms")

    return metrics

##### FAISS #####
# FLAT
def faiss_search_flat(df, top_k=10, test_mode=False):
    embedding_matrix = np.stack(df['embedding'].values).astype(np.float32)

    def build_flat_index(dim, embeddings):
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        return index

    return _evaluate_faiss_index(df, embedding_matrix, build_flat_index, top_k, test_mode)

# IVF
def faiss_search_ivf(df, top_k=10, nlist=100, test_mode=False):
    embedding_matrix = np.stack(df['embedding'].values).astype(np.float32)

    def build_ivf_index(dim, embeddings):
        quantizer = faiss.IndexFlatL2(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist)
        index.train(embeddings)
        index.add(embeddings)
        return index

    return _evaluate_faiss_index(df, embedding_matrix, build_ivf_index, top_k, test_mode)

# HNSW
def faiss_search_hnsw(df, top_k=10, m=32, test_mode=False):
    embedding_matrix = np.stack(df['embedding'].values).astype(np.float32)

    def build_hnsw_index(dim, embeddings):
        index = faiss.IndexHNSWFlat(dim, m)
        index.add(embeddings)
        return index

    return _evaluate_faiss_index(df, embedding_matrix, build_hnsw_index, top_k, test_mode)

# Inner Product 
def faiss_search_innerproduct(df, top_k=10, test_mode=False, normalize=True):
    embedding_matrix = np.stack(df['embedding'].values).astype(np.float32)

    if normalize:
        norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
        embedding_matrix = embedding_matrix / (norms + 1e-10)

    def build_flat_ip_index(dim, embeddings):
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        return index

    return _evaluate_faiss_index(df, embedding_matrix, build_flat_ip_index, top_k, test_mode)

# IVFPQ
def faiss_search_ivfpq(df, top_k=10, nlist=100, m=16, nbits=8, test_mode=False):
    embedding_matrix = np.stack(df['embedding'].values).astype(np.float32)

    def build_ivfpq_index(dim, embeddings):
        quantizer = faiss.IndexFlatL2(dim)
        index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, nbits)
        index.train(embeddings)
        index.add(embeddings)
        return index

    return _evaluate_faiss_index(df, embedding_matrix, build_ivfpq_index, top_k, test_mode)

# IVFPQR
def faiss_search_ivfpqr(df, top_k=10, nlist=100, m=16, nbits=8, nprobe=10, test_mode=False):
    embedding_matrix = np.stack(df['embedding'].values).astype(np.float32)

    def build_ivfpqr_index(dim, embeddings):
        quantizer = faiss.IndexFlatL2(dim)
        pq_index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, nbits)
        pq_index.train(embeddings)
        pq_index.add(embeddings)
        pq_index.nprobe = nprobe

        # Wrap with IndexRefineFlat (DO NOT ADD AGAIN)
        index = faiss.IndexRefineFlat(pq_index)

        return index

    return _evaluate_faiss_index(df, embedding_matrix, build_ivfpqr_index, top_k, test_mode)


##### hnswlib #####
def hnswlib_search(df, efc=200, m=32, efs=200, top_k=10, test_mode=False):
    embedding_matrix = np.stack(df['embedding'].values).astype(np.float32)
    dim = embedding_matrix.shape[1]

    total = 0
    top_k_metrics = {1: 0, 3: 0, 10: 0}
    top_k_failures = {1: [], 3: [], 10: []}
    search_time = []

    if test_mode:
        print("##### test mode #####")

    for query_idx in tqdm(range(len(df)), desc="HNSWLIB Exclude-Self Search"):
        if test_mode and query_idx >= 100:
            break

        query_id = df.iloc[query_idx]['ID']
        query_name = df.iloc[query_idx]['name']
        query_vector = embedding_matrix[query_idx].reshape(1, -1)

        if df[df.ID == query_id].shape[0] == 1:
            continue

        total += 1

        # Remove query from dataset
        df_copy = df.drop(index=query_idx).reset_index(drop=True)
        embedding_matrix_drop = np.stack(df_copy['embedding'].values).astype(np.float32)

        # Rebuild HNSW index (new for each query)
        index = hnswlib.Index(space='cosine', dim=dim)
        index.init_index(max_elements=len(embedding_matrix_drop), ef_construction=efc, M=m)
        index.set_ef(efs)
        index.add_items(embedding_matrix_drop, np.arange(len(embedding_matrix_drop)))

        start_time = datetime.now()
        labels, distances = index.knn_query(query_vector, k=top_k)
        end_time = datetime.now()

        search_time.append((end_time - start_time).total_seconds())

        top_k_indices = labels[0]
        top_k_ids = [df_copy.iloc[i]['ID'] for i in top_k_indices]

        for k in [1, 3, 10]:
            if evaluate_topk(query_id, top_k_ids, k):
                top_k_metrics[k] += 1
            else:
                top_k_failures[k].append(log_topk_failure(query_idx, query_id, query_name, top_k_ids, k))

    # Final reporting
    metrics = {f'top{k}_acc': top_k_metrics[k] / total for k in [1, 3, 10]}
    metrics.update({f'top{k}_failures': top_k_failures[k] for k in [1, 3, 10]})
    metrics['median_latency_ms'] = np.median(search_time) * 1000
    metrics['index_type'] = 'hnswlib'
    metrics['ef_construction'] = efc
    metrics['ef_search'] = efs
    metrics['M'] = m

    print(f"Total queries evaluated: {total}")
    for k in [1, 3, 10]:
        print(f"Top-{k} Accuracy: {metrics[f'top{k}_acc']:.4f}")
        print(f"Top-{k} Failures: {len(metrics[f'top{k}_failures'])}")
    print(f"Median Query Time: {metrics['median_latency_ms']:.2f} ms")

    return metrics

##### brute force #####
def brute_force_search(df, detector, top_k=10, test_mode=False):
    top_k_metrics = {1: 0, 3: 0, 10: 0}
    top_k_failures = {1: [], 3: [], 10: []}
    search_time = []
    total = 0

    if test_mode:
        print("##### test mode #####")

    for query_idx in tqdm(range(len(df)), desc="Brute Force Search"):
        if test_mode and query_idx >= 100:
            break

        query_emb = df['embedding'].iloc[query_idx]
        query_id = df.iloc[query_idx]['ID']
        query_name = df.iloc[query_idx]['name']

        # Skip if only one sample of this ID
        if df[df.ID == query_id].shape[0] == 1:
            continue
        total += 1

        start_time = datetime.now()

        # Compute similarity to all others (excluding self)
        similarities = []
        for i in range(len(df)):
            if i == query_idx:
                continue
            emb = df['embedding'].iloc[i]
            sim = detector.compute_sim(query_emb, emb)
            similarities.append((i, sim))
        
        end_time = datetime.now()
        search_time.append((end_time - start_time).total_seconds())

        # Sort by similarity descending
        sorted_indices = [i for i, _ in sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]]
        top_k_ids = [df.iloc[i]['ID'] for i in sorted_indices]

        # Evaluate top-k levels
        for k in [1, 3, 10]:
            if evaluate_topk(query_id, top_k_ids, k):
                top_k_metrics[k] += 1
            else:
                top_k_failures[k].append(log_topk_failure(query_idx, query_id, query_name, top_k_ids, k))

        

    # Final reporting
    metrics = {f'top{k}_acc': top_k_metrics[k] / total for k in [1, 3, 10]}
    metrics.update({f'top{k}_failures': top_k_failures[k] for k in [1, 3, 10]})
    metrics['median_latency_ms'] = np.median(search_time) * 1000
    metrics['index_type'] = 'brute_force'

    print(f"Total queries evaluated: {total}")
    for k in [1, 3, 10]:
        print(f"Top-{k} Accuracy: {metrics[f'top{k}_acc']:.4f}")
        print(f"Top-{k} Failures: {len(metrics[f'top{k}_failures'])}")
    print(f"Median Query Time: {metrics['median_latency_ms']:.2f} ms")

    return metrics


##### fvs #####
import fvs_face_helper as fvs
import importlib
importlib.reload(fvs)

def fvs_search(df, test_mode=False, top_k=10):
    total = 0
    top_k_metrics = {1: 0, 3: 0, 10: 0}
    top_k_failures = {1: [], 3: [], 10: []}
    search_time = []

    if test_mode:
        print("##### test mode #####")

    for query_idx in tqdm(range(len(df)), desc="FVS Search"):
        if test_mode and query_idx >= 100:
            break

        query_id = df.iloc[query_idx]['ID']
        query_name = df.iloc[query_idx]['name']
        query_emb = df['embedding'].iloc[query_idx]

        if df[df.ID == query_id].shape[0] == 1:
            continue  # Skip singletons

        total += 1
        query_emb = query_emb.reshape((1, -1))

        start_time = datetime.now()
        response = fvs.face_search("56d7e7b2-4c17-47a9-b247-3710120b5466", query_emb, top_k, verbose=False)
        end_time = datetime.now()
        search_time.append((end_time - start_time).total_seconds())

        top_k_indices = response.indices[0][1:]  # Skip top-1 (self-match)
        top_k_scores = [float(x) for x in response.distance[0][1:]]
        top_k_ids = [int(df.iloc[i]['ID']) for i in top_k_indices]

        # Top-1 Evaluation
        if top_k_ids[0] == query_id:
            top_k_metrics[1] += 1
        else:
            top_k_failures[1].append({
                'query_idx': query_idx,
                'query_id': query_id,
                'query_name': query_name,
                'top_k_ids': top_k_ids[:1],
                'top_k_scores': top_k_scores[:1],
                'top_k_names': [df.iloc[i]['name'] for i in top_k_indices[:1]]
            })

        # Top-3 Evaluation
        if query_id in top_k_ids[:3]:
            top_k_metrics[3] += 1
        else:
            top_k_failures[3].append({
                'query_idx': query_idx,
                'query_id': query_id,
                'query_name': query_name,
                'top_k_ids': top_k_ids[:3],
                'top_k_scores': top_k_scores[:3],
                'top_k_names': [df.iloc[i]['name'] for i in top_k_indices[:3]]
            })

        # Top-10 Evaluation
        if query_id in top_k_ids[:10]:
            top_k_metrics[10] += 1
        else:
            top_k_failures[10].append({
                'query_idx': query_idx,
                'query_id': query_id,
                'query_name': query_name,
                'top_k_ids': top_k_ids,
                'top_k_scores': top_k_scores,
                'top_k_names': [df.iloc[i]['name'] for i in top_k_indices]
            })

    # Final Reporting
    metrics = {
        'index_type': 'fvs',
        'top1_acc': top_k_metrics[1] / total,
        'top3_acc': top_k_metrics[3] / total,
        'top10_acc': top_k_metrics[10] / total,
        'median_latency_ms': np.median(search_time) * 1000,
        'top1_failures': top_k_failures[1],
        'top3_failures': top_k_failures[3],
        'top10_failures': top_k_failures[10]
    }

    print(f"Total queries evaluated: {total}")
    for k in [1, 3, 10]:
        print(f"Top-{k} Accuracy: {metrics[f'top{k}_acc']:.4f}")
        print(f"Top-{k} Failures: {len(metrics[f'top{k}_failures'])}")
    print(f"Median Query Time: {metrics['median_latency_ms']:.2f} ms")

    return metrics