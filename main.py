
# Requires transformers>=4.51.0

import torch
import torch.nn.functional as F

from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from llm_models import QwenInstruct, QwenReasoning

from configs import parse_args

VALID_DATASET_NAMES = ["huyen_research_abstracts"]

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



def process_pandas_data(dataset_name, verbose=False):
    """
    This function is intended to process given a given dataset into a pandas dataframe
    Ideally I want to make it more modular, but this is limited by uniqueness between data sources
    Eventually I want to extend it to work with scraping text from single sources like large PDFs 
    """
    assert dataset_name in VALID_DATASET_NAMES, print(f"Data source {dataset_name} is not implemented for use with Pandas")
    df = None

    if dataset_name == "huyen_research_abstracts":
        ### Import the data for the abstracts to be used a sentences
        datapath = os.path.join(".", "data", dataset_name)
        df = pd.read_csv(datapath)
        
        df = df.drop("ai_generated", axis=1)
        df = df.sort_values(by=["is_ai_generated", "title"]).reset_index(drop=True)

        df["title_embeddings"] = [None]*df.shape[0]
        df["abstract_embeddings"] = [None]*df.shape[0]
        
        # Each query must come with a one-sentence instruction that describes the task
        task = 'Given a research paper title query, retrieve the abstract that best fits the title.'
        max_length, batch_size = 3192, 64

        titles = df["title"].tolist()
        abstracts = df["abstract"].tolist()
        print(f"Running {len(titles)//batch_size} batches")
        for batch in range((len(titles)//batch_size)+1):
            start_idx, end_idx = (batch * batch_size), min((batch+1)*batch_size, len(titles))

            for step, input_text in [("title_embeddings", titles), ("abstract_embeddings", abstracts)]:
                if step == "title_embeddings":
                    batch_text = [get_detailed_instruct(task, title) for title in titles[start_idx:end_idx]]
                else:
                    # No need to add instruction for retrieval documents
                    batch_text = abstracts[start_idx:end_idx]

                embeddings = embed_batch(batch_text)

                if verbose:
                    print(f"Inserting embeddings with shape {embeddings.shape} in df idx: {start_idx}")
                for i in range(embeddings.shape[0]):
                    df.at[start_idx + i, step] = (embeddings[i].cpu().float().numpy())
    return df



def process_faiss_data(dataframe, dataset_name, index_type = "flatip", verbose=False):
    """
    Make a FAISS index db for the given dataset using the pandas dataframe
    Since different datasets have different document names (e.g. Abstracts for huyen), we do this based on the dataset name.
    """
    assert dataset_name in VALID_DATASET_NAMES, print(f"Data source {dataset_name} is not implemented for use with FAISS")


    if dataset_name == "huyen_research_abstracts":
        ### Set up FAISS vector DB
        embeddings = np.stack(dataframe["abstract_embeddings"].values)
    
    if index_type == "flatip":
        faiss_db = faiss.IndexFlatIP(embeddings.shape[1])
    
    faiss_db.add(abstract_embeddings)
    
    index_path = "./data/{dataset_name}.faiss"
    faiss.write_index(faiss_db, index_path)


def process_chroma_data(dataframe, dataset_name, collection_type, verbose=False):
    """
    Make a Chroma index db for the given dataset using the pandas dataframe
    Since different datasets have different document names (e.g. Abstracts for huyen), we do this based on the dataset name.
    """
    assert dataset_name in VALID_DATASET_NAMES, print(f"Data source {dataset_name} is not implemented for use with ChromaDB")

    client = chromadb.PersistentClient(path="./data/chromadb_{dataset_name}")

    ### Check whether the collection already exists
    existing_collections = [c.name for c in client.list_collections()]
    if dataset_name == "huyen_research_abstracts":
        collection_name = "paper_abstracts"


    if collection_name in existing_collections:
        collection = client.get_collection(collection_name)

        if verbose:
            print(f"Chroma collection '{collection_name}' already exists. Skipping document insertion.")



    else:
        ### Create the specified collection type
        if collection_type == "hnsw_space_cosine":
            collection = client.create_collection(
                name=collection_name,
                metadata={
                    "hnsw:space": "cosine"
                }
            )



        ### Populate the collection with documents matching the data source being used
        if dataset_name == "huyen_research_abstracts":
            collection.add(
                ids=[f"paper_{i}" for i in range(len(df))],
                documents=df["abstract"].tolist(),
                embeddings=df["abstract_embeddings"].tolist(),
                metadatas=[
                    {
                        "title": row["title"],
                        "is_ai_generated": bool(row["is_ai_generated"])
                    }
                    for _, row in df.iterrows()
                ]
            )



def prepare_data(dataset_name, databases: list[str] = ["pandas", "faiss", "chromadb"], index_type = "flatip", collection_type = "hnsw_space_cosine", verbose=False):
    """
    For a specified data source, produce the vector databases that will be used for later RAG operations
    Note: Currently the type of documents stored in a given database are dependent on the data source in a 1:1 manner
            This means that each source is constructs a single possible document type.
            This may need to be changed if we want sources that produce multiple document types. 
            Alternatively they would need to be treated as distinct dataset names using the same data source.
    """

    if "pandas" in databases:
        ### Each valid dataset is handled according to its text source structure then saved as a dataframe
        df = process_pandas_data(dataset_name, verbose)
        df.to_parquet("./data/{dataset_name}.parquet")
    else:
        assert os.isfile("./data/{dataset_name}.parquet"), print("Pandas data needs to be prepared prior to setting up other databases")
        df = pandas.read_parquet(f"./data/{dataset_name}.parquet")


    if "faiss" in databases:
        process_faiss_data(df, dataset_name, index_type, verbose)

    if "chromadb" in databases:
        process_chroma_data(df, dataset_name, collection_type, verbose)




def main():

    args = parse_args()

    embedder_tokenizer = AutoTokenizer.from_pretrained(args.embedding_model_name, padding_side='left').to(args.device)
    embedder = AutoModel.from_pretrained(args.embedding_model_name).to(args.device)


    if args.operation == "data_preparation":
        prepare_data(
            args.text_data_source, 
            databases = ["pandas", "faiss", "chromadb"],
            index_type = args.faiss_index_type, 
            verbose=args.verbose
        )







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



