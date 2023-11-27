import os
import datetime
from langchain import document_loaders
from langchain.docstore.document import Document
from langchain.embeddings import GPT4AllEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All
from langchain import hub
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter


def load_docs(directory: str) -> list[Document]:
    text_loader_args={"autodetect_encoding": True, "mode": "elements"}
    docs = []
    docs = document_loaders.DirectoryLoader(path="../data", glob="**/*.json", recursive=True, show_progress=True, silent_errors=True, loader_cls=document_loaders.JSONLoader, loader_kwargs={"jq_schema": ".", "text_content": False}).load()
    docs += document_loaders.DirectoryLoader(path="../data", glob="**/*.csv", recursive=True, show_progress=True, silent_errors=True, loader_cls=document_loaders.CSVLoader, loader_kwargs={"csv_args":{"delimiter": "\n"}}).load()
    # load all non-json/csv files as text
    docs += document_loaders.DirectoryLoader(path=directory, glob="**/*[!(.json|.csv)]", recursive=True, show_progress=True, silent_errors=True, loader_kwargs=text_loader_args).load()
    return docs


docs = load_docs("../data")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(docs)

def store_documents(documents: list[Document], embedding) -> Chroma:
    db = Chroma.from_documents(documents=documents, embedding=embedding, persist_directory="../chromadb/", collection_name="personal_assistant_new_embedding", collection_metadata={"timestamp": datetime.datetime.now().isoformat()})
    return db

db = store_documents(all_splits, GPT4AllEmbeddings())

query = "What category is Transport of London in?"

# similarity search using db
# returned_docs = d.similarity_search(query)
# print(returned_docs[:10].page_content)

llm = GPT4All(
    model="../models/gpt4all/gpt4all-falcon-q4_0.gguf",
    max_tokens=2048,
)

rag_prompt = hub.pull("rlm/rag-prompt")

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=db.as_retriever(),
    chain_type_kwargs={"prompt": rag_prompt},
)

qa_chain({"query": query})
