import os
from langchain import document_loaders
from langchain.docstore.document import Document
from langchain.embeddings import GPT4AllEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All
from langchain import hub
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_data(directory: str) -> list[Document]:
    extension_with_loader = {
        "csv": document_loaders.CSVLoader,
        "txt": document_loaders.TextLoader,
        "html": document_loaders.BSHTMLLoader,
        "json": document_loaders.JSONLoader,
        "md": document_loaders.UnstructuredMarkdownLoader,
    }
    data = []

    for extension, loader in extension_with_loader.items():
        print(f"Loading {extension} files from {directory}")
        text_loader_kwargs={'autodetect_encoding': True}
        data += document_loaders.DirectoryLoader(path=directory, glob=f"**/*.{extension}", recursive=True, loader_cls=loader, show_progress=True, use_multithreading=True, loader_kwargs=text_loader_kwargs).load()
        print(f"Loaded {len(data)} {extension} files successfully!")

    print(f"Loaded all {len(data)} files successfully!")
    return data

docs = load_data("../data")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(docs)

vectorstore = Chroma.from_documents(documents=all_splits, embedding=GPT4AllEmbeddings())

llm = GPT4All(
    model="../models/gpt4all/gpt4all-falcon-q4_0.gguf",
    max_tokens=2048,
)

rag_prompt = hub.pull("rlm/rag-prompt")

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectorstore.as_retriever(),
    chain_type_kwargs={"prompt": rag_prompt},
)

question = "What category is Transport of London in?"
qa_chain({"query": question})
