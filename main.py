
# Requires transformers>=4.51.0

import torch
import torch.nn.functional as F

from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from llm_models import QwenInstruct, QwenReasoning

from configs import parse_args




def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery:{query}'





def main():


    args = parse_args()

    embedder_tokenizer = AutoTokenizer.from_pretrained(args.embedding_model_name, padding_side='left').to(args.device)
    embedder = AutoModel.from_pretrained(args.embedding_model_name).to(args.device)


    # Tokenize the input texts
    batch_dict_queries = embedder_tokenizer(
        queries,
        padding=True,
        truncation=True,
        max_length=args.max_tokens,
        return_tensors="pt",
    )
    # Tokenize the input texts
    batch_dict_documents = embedder_tokenizer(
        documents,
        padding=True,
        truncation=True,
        max_length=args.max_tokens,
        return_tensors="pt",
    )

    batch_dict_queries.to(args.device)
    batch_dict_documents.to(args.device)
    with torch.no_grad():
        start_time = time.time()
        outputs = embedder(**batch_dict_queries)
        query_embeddings = last_token_pool(outputs.last_hidden_state, batch_dict_queries['attention_mask'])
        # normalize embeddings
        query_embeddings = F.normalize(query_embeddings, p=2, dim=1)

        print("Time taken to embed queries: ", time.time() - start_time)
        start_time = time.time()

        outputs = embedder(**batch_dict_documents)
        document_embeddings = last_token_pool(outputs.last_hidden_state, batch_dict_documents['attention_mask'])
        document_embeddings = F.normalize(document_embeddings, p=2, dim=1)
        print("Time taken to embed documents: ", time.time() - start_time)

        scores = (query_embeddings @ document_embeddings.T)

    print(scores.tolist())









if __name__ == "__main__":

    main()



