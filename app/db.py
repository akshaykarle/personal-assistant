import datetime
from typing import Optional

from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.vectorstores.utils import filter_complex_metadata
from langchain_core.vectorstores import VectorStoreRetriever


class Database:
    def __init__(self, collection_name: Optional[str] = "personal_assistant", embeddings_model_name: Optional[str] = "all-MiniLM-L6-v2"):
        embedding = HuggingFaceEmbeddings(model_name=embeddings_model_name)
        self.docstore = Chroma(collection_name=collection_name, embedding_function=embedding, persist_directory="../chromadb/", collection_metadata={"timestamp": datetime.datetime.now().isoformat()})

    def is_empty(self):
        return len(self.docstore.get()['documents']) == 0

    def as_retriever(self, search_type: Optional[str], search_kwargs: Optional[dict]) -> VectorStoreRetriever:
        return self.docstore.as_retriever(search_type, search_kwargs)

    def store_documents(self, documents: list[Document]):
        print(f"Storing {len(documents)} documents into db")
        filtered_docs = filter_complex_metadata(documents)
        self.docstore.add_documents(filtered_docs, show_progress=True)
