import argparse
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--llm_model_name", type=str, default="Qwen/Qwen3-Embedding-0.6B", choices=["Qwen/Qwen3-Embedding-0.6B"], help="Name of LLM model to use")
    parser.add_argument("--embedding_model_name", type=str, default="Qwen/Qwen3-Embedding-0.6B", choices=["Qwen/Qwen3-Embedding-0.6B"], help="Name of Embedding model to use")

    parser.add_argument("--top_k", type=int, default=3, help="Number of documents to return with RAG")
    parser.add_argument("--max_tokens", type=int, default=3000, help="Maximum number of tokens for prompts")



    parser.add_argument("--verbose", action="store_true", help="Provide verbose outputs of various intermediate steps")


    args = parser.parse_args()

    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    return args
