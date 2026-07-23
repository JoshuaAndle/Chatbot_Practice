
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


import faiss
import chromadb





def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]



def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery:{query}'






class DataBaseManager():
    def __init__(
            self,
            embedder_name: str,
            dataset_name: str, 
            index_type: str, 
            collection_type: str, 
            max_tokens: int, 
            db_batch_size: int, 
            verbose: bool,
            device: str
        ):

        ### Prepare the embedders to be used for vector database setup and retrieval
        self.embedder_tokenizer = AutoTokenizer.from_pretrained(embedder_name, padding_side='left')
        self.embedder = AutoModel.from_pretrained(embedder_name).to(device)
        self.device = device

        self.dataset_name = dataset_name
        self.index_type = index_type
        self.collection_type = collection_type
        self.verbose = verbose
        self.max_tokens = max_tokens
        self.db_batch_size = db_batch_size
        self.df = None
        self.task_description = self.get_task_description()


    def get_task_description(self):
        if self.dataset_name == "huyen_research_abstracts":
            self.task_description = 'Given a research paper title query, retrieve the abstract that best fits the title.'

        else:
            raise NotImplementedError(f"{self.dataset_name} not implemented for task instructions")

    def embed_batch(self, batch_text: list[str]) -> Tensor:
        """
        Embeds the given batch of text strings. This is currently setup for Qwen3-Embedder, API may need to change when others are added
        """
        # Tokenize the input abstracts
        batch_dict = self.embedder_tokenizer(
            batch_text,
            padding=True,
            truncation=True,
            max_length=self.max_tokens,
            return_tensors="pt",
        )

        batch_dict = batch_dict.to(self.device)

        with torch.no_grad():
            start_time = time.time()

            outputs = self.embedder(**batch_dict)
            embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            embeddings = F.normalize(embeddings, p=2, dim=1).to("cpu")
            print("Time taken to embed documents: ", time.time() - start_time)
            
            return embeddings




    def process_pandas_data(self):
        """
        This function is intended to process given a given dataset into a pandas dataframe
        Ideally I want to make it more modular, but this is limited by uniqueness between data sources
        Eventually I want to extend it to work with scraping text from single sources like large PDFs 
        """
        df = None

        if self.dataset_name == "huyen_research_abstracts":
            ### Import the data for the abstracts to be used a sentences
            datapath = os.path.join(".", "data", self.dataset_name + ".csv")
            df = pd.read_csv(datapath)
            
            df = df.drop("ai_generated", axis=1)
            df = df.sort_values(by=["is_ai_generated", "title"]).reset_index(drop=True)

            df["title_embeddings"] = [None]*df.shape[0]
            df["abstract_embeddings"] = [None]*df.shape[0]
            
            # Each query must come with a one-sentence instruction that describes the task
            # task = 'Given a research paper title query, retrieve the abstract that best fits the title.'
            batch_size = self.db_batch_size

            titles = df["title"].tolist()
            abstracts = df["abstract"].tolist()
            print(f"Running {len(titles)//batch_size} batches")
            for batch in range((len(titles)//batch_size)+1):
                start_idx, end_idx = (batch * batch_size), min((batch+1)*batch_size, len(titles))

                for step, input_text in [("title_embeddings", titles), ("abstract_embeddings", abstracts)]:
                    if step == "title_embeddings":
                        batch_text = [get_detailed_instruct(self.task_description, title) for title in titles[start_idx:end_idx]]
                    else:
                        # No need to add instruction for retrieval documents
                        batch_text = abstracts[start_idx:end_idx]

                    embeddings = self.embed_batch(batch_text)

                    if self.verbose:
                        print(f"Inserting embeddings with shape {embeddings.shape} in df idx: {start_idx}")
                    for i in range(embeddings.shape[0]):
                        df.at[start_idx + i, step] = (embeddings[i].cpu().float().numpy())
            self.df = df


    def process_faiss_data(self):
        """
        Make a FAISS index db for the given dataset using the pandas dataframe
        Since different datasets have different document names (e.g. Abstracts for huyen), we do this based on the dataset name.
        """

        assert self.df is not None, "Trying to process faiss without loading pandas dataframe"

        if self.dataset_name == "huyen_research_abstracts":
            ### Set up FAISS vector DB
            embeddings = np.stack(self.df["abstract_embeddings"].values)
        
        if self.index_type == "flatip":
            faiss_db = faiss.IndexFlatIP(embeddings.shape[1])
        
        faiss_db.add(embeddings)

        index_path = f"./data/{self.dataset_name}.faiss"
        faiss.write_index(faiss_db, index_path)


    def process_chroma_data(self):
        """
        Make a Chroma index db for the given dataset using the pandas dataframe
        Since different datasets have different document names (e.g. Abstracts for huyen), we do this based on the dataset name.
        """

        client = chromadb.PersistentClient(path=f"./data/chromadb_{self.dataset_name}")

        assert self.df is not None, "Trying to process chroma database without loading pandas dataframe"

        ### Check whether the collection already exists
        existing_collections = [c.name for c in client.list_collections()]
        if self.dataset_name == "huyen_research_abstracts":
            collection_name = "paper_abstracts"


        if collection_name in existing_collections:
            collection = client.get_collection(collection_name)

            if self.verbose:
                print(f"Chroma collection '{collection_name}' already exists. Skipping document insertion.")



        else:
            ### Create the specified collection type
            if self.collection_type == "hnsw_space_cosine":
                collection = client.create_collection(
                    name=collection_name,
                    metadata={
                        "hnsw:space": "cosine"
                    }
                )



            ### Populate the collection with documents matching the data source being used
            if self.dataset_name == "huyen_research_abstracts":
                collection.add(
                    ids=[f"paper_{i}" for i in range(len(self.df))],
                    documents=self.df["abstract"].tolist(),
                    embeddings=self.df["abstract_embeddings"].tolist(),
                    metadatas=[
                        {
                            "title": row["title"],
                            "is_ai_generated": bool(row["is_ai_generated"])
                        }
                        for _, row in self.df.iterrows()
                    ]
                )


    def prepare_data(self, databases: list[str] = ["pandas", "faiss", "chromadb"]):
        """
        For a specified data source, produce the vector databases that will be used for later RAG operations
        Note: Currently the type of documents stored in a given database are dependent on the data source in a 1:1 manner
                This means that each source is constructs a single possible document type.
                This may need to be changed if we want sources that produce multiple document types. 
                Alternatively they would need to be treated as distinct dataset names using the same data source.
        """

        if "pandas" in databases:
            ### Each valid dataset is handled according to its text source structure then saved as a dataframe
            self.process_pandas_data()
            self.df.to_parquet(f"./data/{self.dataset_name}.parquet")
        else:
            assert os.isfile(f"./data/{self.dataset_name}.parquet"), "Pandas data needs to be prepared prior to setting up other databases"
            self.df = pandas.read_parquet(f"./data/{self.dataset_name}.parquet")


        if "faiss" in databases:
            self.process_faiss_data()

        if "chromadb" in databases:
            self.process_chroma_data()


    def query_database(self, queries: list[str], top_k: int):
        pass

    def load_database(self):
        pass




class Pandas_DataBaseManager(DataBaseManager):
    """
    This class is designed as a 'scratch' implementation of a vector database in Pandas.
    It is fairly simple and only handles sequential querying of all samples rather than ANN
    """
    def __init__(
            self,
            embedder_name: str,
            dataset_name: str, 
            index_type: str, 
            collection_type: str, 
            max_tokens: int, 
            db_batch_size: int, 
            verbose: bool,
            device: str
        ):
        super(Pandas_DataBaseManager, self).__init__(embedder_name, dataset_name, index_type, collection_type, max_tokens, db_batch_size, verbose, device)



    def load_database(self):
        pandas_filepath = f"./data/{self.dataset_name}.parquet"
        assert os.path.isfile(pandas_filepath), f"Pandas Database not found at {pandas_filepath}. Need to run data_preparation operation before using database."
        self.df = pd.read_parquet(pandas_filepath)


    def query_database(self, queries: list[str], top_k: int, embedding_type: str = "abstract"):

        if self.dataset_name == "huyen_research_abstracts":
            if embedding_type == "abstract":
                embedding_column = "abstract_embeddings"
                document_column = "abstract"
            elif embedding_type == "title":
                embedding_column = "title_embeddings"
                document_column = "title"
            else:
                raise ValueError("Only abstract and title embeddings are implemented for Pandas DB")

            batch_text = [get_detailed_instruct(self.task_description, title) for title in queries]
            query_embeddings = self.embed_batch(batch_text)
            query_embeddings = query_embeddings.cpu().float().numpy()


            document_embeddings = np.stack(self.df[embedding_column].values)
            scores = np.dot(query_embeddings, document_embeddings.T)


        matched_documents = []
        ### Get topk indices for each query as 2d array
        top_indices = np.argsort(scores)[:,-top_k:]
        for q in range(len(top_indices)):
            print("-"*100,f"\nMatches for query: {queries[q]}")            
            matched_documents.append(self.df[document_column][top_indices[q]].values.astype("U"))

            for k in top_indices[q]:
                print(k, scores[q, k], self.df.iloc[k][document_column])


        return scores, top_indices, np.array(matched_documents)







class FAISS_DataBaseManager(DataBaseManager):
    def __init__(
            self,
            embedder_name: str,
            dataset_name: str, 
            index_type: str, 
            collection_type: str, 
            max_tokens: int, 
            db_batch_size: int, 
            verbose: bool,
            device: str
        ):
        super(FAISS_DataBaseManager, self).__init__(embedder_name, dataset_name, index_type, collection_type, max_tokens, db_batch_size, verbose, device)



    def load_database(self):
        ### Note: FAISS still relies on pandas df to lookup the indexed documents
        pandas_filepath = f"./data/{self.dataset_name}.parquet"
        faiss_filepath = f"./data/{self.dataset_name}.faiss"
        assert os.path.isfile(pandas_filepath), f"Pandas Database not found at {pandas_filepath}. Need to run data_preparation operation before using database."
        assert os.path.isfile(faiss_filepath), f"FAISS Database not found at {faiss_filepath}. Need to run data_preparation operation before using database."
        self.df = pd.read_parquet(pandas_filepath)

        if self.index_type == "flatip":
            self.index = faiss.read_index(faiss_filepath)
        else:
            raise NotImplementedError("FAISS Index types other than flatip have not been implemented yet")



    def query_database(self, queries: list[str], top_k: int, embedding_type: str = "abstract"):

        ### Get the document results from the query embeddings
        batch_text = [get_detailed_instruct(self.task_description, title) for title in queries]
        query_embeddings = self.embed_batch(batch_text)
        query_embeddings = query_embeddings.cpu().float().numpy()

        scores, indices = self.index.search(query_embeddings, top_k)

        ### Look up the appropriate documents for the top indices using pandas dataframe
        if self.dataset_name == "huyen_research_abstracts":
            if embedding_type == "abstract":
                embedding_column = "abstract_embeddings"
                document_column = "abstract"
            elif embedding_type == "title":
                embedding_column = "title_embeddings"
                document_column = "title"
            else:
                raise ValueError("Only abstract and title embeddings are implemented for Pandas DB")

        matched_documents = []
        ### Get topk indices for each query as 2d array
        for q in range(query_embeddings.shape[0]):
            print("-"*100,f"\nMatches for query {queries[q]}")
            matched_documents.append(self.df[document_column][indices[q]].values.astype("U"))

            for i, k in enumerate(indices[q]):
                print(f"Match {i} for document {k} score: {scores[q][i]} and document:\n {self.df[document_column][k]}")

        return scores, indices, np.array(matched_documents)


class Chroma_DataBaseManager(DataBaseManager):
    def __init__(
            self,
            embedder_name: str,
            dataset_name: str, 
            index_type: str, 
            collection_type: str, 
            max_tokens: int, 
            db_batch_size: int, 
            verbose: bool,
            device: str
        ):
        super(Chroma_DataBaseManager, self).__init__(embedder_name, dataset_name, index_type, collection_type, max_tokens, db_batch_size, verbose, device)

        if dataset_name == "huyen_research_abstracts":
            self.collection_name = "paper_abstracts"
        else:
            raise NotImplementedError("No Chroma collection is implemented for dataset_name: ", dataset_name)

        self.client = None
        self.collection = None


    def load_database(self):
        chroma_filepath = f"./data/chromadb_{self.dataset_name}"
        assert os.path.isdir(chroma_filepath), f"Chroma Database not found at {chroma_filepath}. Need to run data_preparation operation before using database."
        self.client = chromadb.PersistentClient(path=chroma_filepath)

        if self.collection_type == "hnsw_space_cosine":
            self.collection = self.client.get_collection(name=self.collection_name)
        else:
            raise NotImplementedError(f"Collection type {self.collection_type} not implemented for Chroma collection")

    def query_database(self, queries: list[str], top_k: int):
                

        ### Get the document results from the query embeddings
        batch_text = [get_detailed_instruct(self.task_description, title) for title in queries]
        query_embeddings = self.embed_batch(batch_text)
        query_embeddings = query_embeddings.cpu().float().numpy()

        matches = self.collection.query(query_embeddings=query_embeddings, n_results=top_k)
        
        for q in range(query_embeddings.shape[0]):
            print("-"*100,f"\nMatches for query {queries[q]}")
            for i in range(top_k):
                print(f"Match {i} found document {matches["ids"][q][i]} with score: {matches["distances"][q][i]} and document:\n {matches["documents"][q][i]}")

        return matches["distances"], matches["ids"], np.array(matches["documents"])





