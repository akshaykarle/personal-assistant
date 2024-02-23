from typing import Optional

import chromadb
from chromadb.utils import embedding_functions


class Database:
    def __init__(self, collection_name: Optional[str] = "personal_assistant"):
        self.chroma_client = chromadb.PersistentClient(path="../chromadb")
        self.sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        self.collection = self.chroma_client.get_or_create_collection(name=collection_name, embedding_function=self.sentence_transformer_ef)


    def store_document(self, document: str, metadata: dict, id: str):
        self.store_documents(documents=[document], metadatas=[metadata], ids=[id])


    def store_documents(self, documents: list[str], metadatas: list[dict], ids: list[str]):
        self.collection.add(documents=documents, metadatas=metadatas, ids=ids)


    def retrieve_documents(self, query: str):
        return self.collection.query(
            query_texts=query,
            n_results=2
        )

    def get_all_documents(self):
        return self.collection.get()


    def reset(self, collection_name: Optional[str] = "personal_assistant"):
        self.chroma_client.delete_collection(collection_name)
        self.collection = self.chroma_client.create_collection(name=collection_name, embedding_function=self.sentence_transformer_ef)
