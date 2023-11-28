from langchain.document_loaders import CSVLoader, DirectoryLoader, JSONLoader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter


def split_documents(documents: list[Document]) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=50)
    all_splits = text_splitter.split_documents(documents)
    print(f"Succesfully split {len(documents)} documents into {len(all_splits)} chunks...")
    return all_splits


def load_md_docs(directory: str) -> list[Document]:
    docs = []
    md_docs = DirectoryLoader(path=directory, glob="**/*.md", recursive=True, show_progress=True, silent_errors=True, loader_kwargs={"autodetect_encoding": True, "mode": "elements"}).load()
    headers_to_split_on = [
        ("#", "Heading 1"),
        ("##", "Heading 2"),
        ("###", "Heading 3"),
        ("####", "Heading 4"),
        ("\\n", "Paragraph"),
        ("\\n\\n", "Paragraph"),
    ]
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, return_each_line=False)
    for d in md_docs:
        split_docs = md_splitter.split_text(d.page_content)
        docs += [Document(page_content=doc.page_content, metadata=d.metadata) for doc in split_docs]
    return docs

def load_docs_from_dir(directory: str) -> list[Document]:
    docs = []
    docs = DirectoryLoader(path=directory, glob="**/*.json", recursive=True, show_progress=True, silent_errors=True, loader_cls=JSONLoader, loader_kwargs={"jq_schema": ".", "text_content": False}).load()
    docs += DirectoryLoader(path=directory, glob="**/*.csv", recursive=True, show_progress=True, silent_errors=True, loader_cls=CSVLoader, loader_kwargs={"csv_args":{"delimiter": "\n"}}).load()
    docs += load_md_docs(directory)
    # load all non-json/csv/md files as text
    docs += DirectoryLoader(path=directory, glob="**/*[!(.json|.csv|.md)]", recursive=True, show_progress=True, silent_errors=True, loader_kwargs={"autodetect_encoding": True, "mode": "elements"}).load()
    return split_documents(docs)
