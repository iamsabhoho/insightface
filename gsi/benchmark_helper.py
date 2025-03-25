import numpy as np
import pandas as pd
import faiss
from datetime import datetime
from tqdm import tqdm


# index_type: choose 'flat', 'ivf', or 'hnsw'
def faiss_search(df, top_k=10, index_type='flat', nlist=100, hnsw_m=32, test_mode=False):
    embedding_matrix = np.stack(df['embedding'].values).astype(np.float32)
    dim = embedding_matrix.shape[1]

    top_1_pos, top_3_pos, top_10_pos = 0, 0, 0
    top_1_failures, top_3_failures, top_10_failures = [], [], []
    search_time = []
    total = 0

    if test_mode:
        print("test mode...")
        
    for query_idx in tqdm(range(len(df)), desc=f"FAISS-{index_type.upper()} Exclude-Self Search"):
        if test_mode:
            if query_idx >= 100:
                break
        
        query_person_id = df.iloc[query_idx]['ID']
        query_name = df.iloc[query_idx]['name']
        query_vector = embedding_matrix[query_idx].reshape(1, -1)

        # Skip if only one sample for this person
        if df[df.ID == query_person_id].shape[0] == 1:
            continue

        total += 1

        # Create dataset without the query
        mask = np.ones(len(df), dtype=bool)
        mask[query_idx] = False
        reduced_embeddings = embedding_matrix[mask]
        reduced_df = df[mask].reset_index(drop=True)

        # Build FAISS index
        if index_type == 'flat':
            index = faiss.IndexFlatL2(dim)
        elif index_type == 'ivf':
            quantizer = faiss.IndexFlatL2(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, nlist)
            index.train(reduced_embeddings)
        elif index_type == 'hnsw':
            index = faiss.IndexHNSWFlat(dim, hnsw_m)
        else:
            raise ValueError("Unsupported index_type: choose from 'flat', 'ivf', 'hnsw'")

        index.add(reduced_embeddings)

        start_time = datetime.now()

        _, indices = index.search(query_vector, top_k)

        end_time = datetime.now()
        search_time.append((end_time - start_time).total_seconds())

        top_k_indices = indices[0]
        top_k_ids = reduced_df.iloc[top_k_indices]['ID'].tolist()

        # Evaluation
        if top_k_ids[0] == query_person_id:
            top_1_pos += 1
        else:
            top_1_failures.append({
                'query_idx': query_idx,
                'query_id': query_person_id,
                'query_name': query_name,
                'top_k_ids': top_k_ids[:1]
            })

        if query_person_id in top_k_ids[:3]:
            top_3_pos += 1
        else:
            top_3_failures.append({
                'query_idx': query_idx,
                'query_id': query_person_id,
                'query_name': query_name,
                'top_k_ids': top_k_ids[:3]
            })

        if query_person_id in top_k_ids[:10]:
            top_10_pos += 1
        else:
            top_10_failures.append({
                'query_idx': query_idx,
                'query_id': query_person_id,
                'query_name': query_name,
                'top_k_ids': top_k_ids
            })

    # Final report
    top_1_acc = top_1_pos / total
    top_3_acc = top_3_pos / total
    top_10_acc = top_10_pos / total
    median_latency = np.median(search_time)
    
    print(f"Total queries evaluated: {total}")
    print(f"Top-1 Accuracy: {top_1_acc:.4f}")
    print(f"Top-3 Accuracy: {top_3_acc:.4f}")
    print(f"Top-10 Accuracy: {top_10_acc:.4f}")
    print(f"Top-1 Failures: {len(top_1_failures)}")
    print(f"Top-3 Failures: {len(top_3_failures)}")
    print(f"Top-10 Failures: {len(top_10_failures)}")
    print(f"Median Query Time: {median_latency:.4f} seconds")

    return {
        'top1_acc': top_1_acc,
        'top3_acc': top_3_acc,
        'top10_acc': top_10_acc,
        'median_latency': median_latency,
        'top1_failures': top_1_failures,
        'top3_failures': top_3_failures,
        'top10_failures': top_10_failures
    }