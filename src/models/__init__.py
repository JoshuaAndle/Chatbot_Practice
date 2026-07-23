
from .llm_models import *


def get_llm(args):
    if args.llm_model_name == "Qwen/Qwen3-0.6B":
        return QwenReasoning(max_history = args.max_history)
    if args.llm_model_name == "Qwen/Qwen2.5-0.5B-Instruct":
        return QwenInstruct(max_history = args.max_history)
    else:
        raise NotImplementedError(f"Database_type {args.database_type} is not implemented for DataBaseManager class")