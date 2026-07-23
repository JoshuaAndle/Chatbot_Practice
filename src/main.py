
# Requires transformers>=4.51.0
import os
import time


import torch
from torch import nn
import torch.nn.functional as F
import pandas as pd
import numpy as np

from torch import Tensor
from transformers import AutoTokenizer, AutoModel

from configs import parse_args

import faiss
import chromadb

from databases import get_database_manager
from databases.database_manager import DataBaseManager





def main():

    args = parse_args()

    if args.operation == "retrieval_only":
        assert(len(args.queries) > 0), "You need to provide a list of one or more queries as args for retrieval_only mode."
        print("Queries for retrieval: ", args.queries)

    ### Prepare the manager for handling vector databases and retrieval operations
    db_manager = get_database_manager(args)

    ### Run the requested RAG operations
    if args.operation == "data_preparation":
        db_manager.prepare_data(databases = ["pandas", "faiss", "chromadb"])


    db_manager.load_database()
    if args.operation == "retrieval_only":
        db_manager.query_database(args.queries, args.top_k)

    else:
        raise ValueError(f"Invalid operation requested: {args.operation}.")










if __name__ == "__main__":

    main()



