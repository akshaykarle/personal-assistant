import datetime
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

collection_name = "personal_assistant"


def initialise_db():
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(collection_name=collection_name, embedding_function=embedding, persist_directory="../chromadb/", collection_metadata={"timestamp": datetime.datetime.now().isoformat()})
    return db


def db_empty(db: Chroma):
    return len(db.get()) == 0


def store_documents(documents: list[Document], db: Chroma):
    print(f"Storing {len(documents)} documents into db")
    db.add_documents(documents, show_progress=True)


# generate a Database class with the above functions as methods
class Database:
    def __init__(self):
        self.docstore = initialise_db()

    def store_documents(self, documents: list[Document]):
        store_documents(documents, self.docstore)

    def is_empty(self):
        return db_empty(self.docstore)

    def as_retriever(self):
        return self.docstore.as_retriever()
