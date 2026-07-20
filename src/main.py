
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


from databases.database_manager import DataBaseManager





def main():

    args = parse_args()


    ### Prepare the manager for handling vector databases and retrieval operations
    db_manager = DataBaseManager(
                    embedder_name = args.embedding_model_name,
                    dataset_name = args.dataset_name, 
                    index_type = args.faiss_index_type, 
                    collection_type = args.chroma_collection_type, 
                    max_tokens = args.max_tokens, 
                    db_batch_size = args.db_batch_size, 
                    verbose = args.verbose,
                    device = args.device
                )

    ### Run the requested RAG operations
    if args.operation == "data_preparation":
        db_manager.prepare_data(databases = ["pandas", "faiss", "chromadb"])

    else:
        raise ValueError(f"Invalid operation requested: {args.operation}.")





    # # Tokenize the input texts
    # batch_dict_queries = embedder_tokenizer(
    #     queries,
    #     padding=True,
    #     truncation=True,
    #     max_length=args.max_tokens,
    #     return_tensors="pt",
    # )
    # # Tokenize the input texts
    # batch_dict_documents = embedder_tokenizer(
    #     documents,
    #     padding=True,
    #     truncation=True,
    #     max_length=args.max_tokens,
    #     return_tensors="pt",
    # )

    # batch_dict_queries.to(args.device)
    # batch_dict_documents.to(args.device)
    # with torch.no_grad():
    #     start_time = time.time()
    #     outputs = embedder(**batch_dict_queries)
    #     query_embeddings = last_token_pool(outputs.last_hidden_state, batch_dict_queries['attention_mask'])
    #     # normalize embeddings
    #     query_embeddings = F.normalize(query_embeddings, p=2, dim=1)

    #     print("Time taken to embed queries: ", time.time() - start_time)
    #     start_time = time.time()

    #     outputs = embedder(**batch_dict_documents)
    #     document_embeddings = last_token_pool(outputs.last_hidden_state, batch_dict_documents['attention_mask'])
    #     document_embeddings = F.normalize(document_embeddings, p=2, dim=1)
    #     print("Time taken to embed documents: ", time.time() - start_time)

    #     scores = (query_embeddings @ document_embeddings.T)

    # print(scores.tolist())









if __name__ == "__main__":

    main()



