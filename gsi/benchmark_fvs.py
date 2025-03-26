from benchmark_helper import fvs_search
import platform
import pandas as pd
from datetime import datetime

# load data
filepath = "/mnt/nas2/sabrina/face-gen/embeddings_sabrina.pkl"
df = pd.read_pickle(filepath)
is_array_col = df['embedding'].apply(lambda x: isinstance(x, np.ndarray))

# convert back to float32
df['embedding'] = df['embedding'].apply(lambda x: x.astype(np.float32))

# set params
top_k = 10
test_mode = False

results = []

# collect info
machine_name = platform.node()
timestamp = datetime.now().isoformat()

results_fvs = fvs_search(df, test_mode=False, top_k=top_k)
flat_result.update({
    "index_type": "flat",
    "machine": machine_name,
    "timestamp": timestamp, 
    "faiss_version": faiss_version
})
results.append(flat_result)

all_results = []
all_results.append(results_fvs)

benchmark_df = pd.DataFrame(all_results)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
benchmark_df.to_csv(f"fvs_benchmark_results_{timestamp}.csv", index=False)