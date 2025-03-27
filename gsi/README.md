# Getting Started 
* ```conda create -n <env_name> python=3.10```
* ```pip install -r requirements.txt```
* See below notebooks/run scripts to run.

# Notebooks
* R50_LFW_embedding.ipynb: Uses Insightface to create embeddings using the default buffalo_l model on LFW dataset.
* search.ipynb: currently contains FVS search function.
* demo.ipynb: demo LFW using fiftyone image gallery viewer.
* faiss-gpu.ipynb: for FAISS GPU testing.
* ijb.ipynb: for IJBC dataset. 

# Benchmark Scipts
* for more information: ```python run_benchmark.py --help```