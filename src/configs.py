import argparse
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="")

    ### Models
    parser.add_argument("--llm_model_name", type=str, default="Qwen/Qwen3-0.6B", choices=["Qwen/Qwen3-0.6B", "Qwen/Qwen2.5-0.5B-Instruct"], help="Name of LLM model to use")
    parser.add_argument("--embedding_model_name", type=str, default="Qwen/Qwen3-Embedding-0.6B", choices=["Qwen/Qwen3-Embedding-0.6B"], help="Name of Embedding model to use")


    ### Data Settings
    parser.add_argument("--dataset_name", type=str, default="huyen_research_abstracts", choices=["huyen_research_abstracts"], help="Name of data source to use for populating vector database")
    parser.add_argument("--faiss_index_type", type=str, default="flatip", choices=["flatip"], help="Type of FAISS Index to create")
    parser.add_argument("--chroma_collection_type", type=str, default="hnsw_space_cosine", choices=["hnsw_space_cosine"], help="Type of ChromaDB Collection to create")


    ### Script Execution Details
    parser.add_argument("--operation", type=str, choices=["data_preparation", "RAG", "RAG_FAISS", "RAG_Chroma", "RAG_LangChain"], help="Which task to execute code for")



    ### Hyperparameters
    parser.add_argument("--top_k", type=int, default=3, help="Number of documents to return with RAG")
    parser.add_argument("--max_tokens", type=int, default=3000, help="Maximum number of tokens for prompts")
    parser.add_argument("--db_batch_size", type=int, default=64, help="Batch size used when constructing the embedding vectors for database preparation")


    ### Flags
    parser.add_argument("--verbose", action="store_true", help="Provide verbose outputs of various intermediate steps")


    args = parser.parse_args()

    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    return args
