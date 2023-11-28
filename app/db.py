import datetime
from typing import Optional

from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma


class Database:
    def __init__(self, collection_name: Optional[str] = "personal_assistant", embeddings_model_name: Optional[str] = "all-MiniLM-L6-v2"):
        embedding = HuggingFaceEmbeddings(model_name=embeddings_model_name)
        self.docstore = Chroma(collection_name=collection_name, embedding_function=embedding, persist_directory="../chromadb/", collection_metadata={"timestamp": datetime.datetime.now().isoformat()})

    def is_empty(self):
        return len(self.docstore.get()['documents']) == 0

    def as_retriever(self):
        return self.docstore.as_retriever()

    def store_documents(self, documents: list[Document]):
        print(f"Storing {len(documents)} documents into db")
        self.docstore.add_documents(documents, show_progress=True)
