
from .database_manager import *


def get_database_manager(args):
	if args.database_type == "pandas":
		return Pandas_DataBaseManager(
                    embedder_name = args.embedding_model_name,
                    dataset_name = args.dataset_name, 
                    index_type = args.faiss_index_type, 
                    collection_type = args.chroma_collection_type, 
                    max_tokens = args.max_tokens, 
                    db_batch_size = args.db_batch_size, 
                    verbose = args.verbose,
                    device = args.device
                )