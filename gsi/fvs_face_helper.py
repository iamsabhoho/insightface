#
# imports
#
# system imports
from datetime import datetime
import time
import sys
import traceback
import numpy as np
# installed packages
import swagger_client
from swagger_client.models import *

#
# config
#
# swagger version flag
VERSION = "v1.0"
# path to the query npy file
SEARCH_PATH = "/home/public/face-gen/query_embedding.npy"

def configure(host="localhost", port=7760, alloc="fvs-automation"):
    config = swagger_client.Configuration()
    config.verify_ssl = False
    config.host = f"http://{host}:{port}/{VERSION}"
    api_config = swagger_client.ApiClient(config)
    api_config.default_headers["allocationToken"] = alloc
    return api_config

def face_build(dataset_path, verbose=False):
    raise Exception("Not yet implemented")

def face_search(dataset_id, query, topk, verbose=False):
    if type(query)==type(""):
        pass # we expect this to be a path
    elif type(query)==type(np.array([])):
        np.save(SEARCH_PATH, query)
        query = SEARCH_PATH
    else:
        raise Exception("Invalid query object")
    api_config = configure() #leaving empty for default params
    search_apis = swagger_client.SearchApi(api_config)
    alloc = api_config.default_headers["allocationToken"]
    if verbose: print("%s: starting search" % sys.argv[0])
    response = search_apis.controllers_search_controller_search(
        SearchRequest(allocation_id=alloc, dataset_id=dataset_id, queries_file_path=query, topk=topk),
        alloc)
    if verbose: print("%s: done searching" % sys.argv[0])
    return response


#
# unit test
# 
if __name__ == "__main__":

    response = face_search("56d7e7b2-4c17-47a9-b247-3710120b5466", "/home/public/face-gen/query_embedding.npy", 10, True )
    print("response=", response)
   
    arr = np.load("/home/public/face-gen/query_embedding.npy") 
    response = face_search("56d7e7b2-4c17-47a9-b247-3710120b5466", arr, 10, True )
    print("response=", response)
