import os
from langchain.document_loaders import CSVLoader, DirectoryLoader, JSONLoader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _split_documents(documents: list[Document]) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=25)
    all_splits = text_splitter.split_documents(documents)
    print(f"Succesfully split {len(documents)} documents into {len(all_splits)} chunks...")
    avg_doc_length = lambda documents: sum([len(doc.page_content) for doc in documents])//len(documents)
    avg_char_count_pre = avg_doc_length(documents)
    avg_char_count_post = avg_doc_length(all_splits)
    print(f'Average length among {len(documents)} documents loaded is {avg_char_count_pre} characters.')
    print(f'After the split we have {len(all_splits)} documents more than the original {len(documents)}.')
    print(f'Average length among {len(all_splits)} documents (after split) is {avg_char_count_post} characters.')
    return all_splits


def _load_md_docs(directory: str) -> list[Document]:
    md_docs = DirectoryLoader(path=directory, glob="**/*.md", recursive=True, show_progress=True, silent_errors=True, loader_kwargs={"autodetect_encoding": True, "mode": "elements"}).load()
    # headers_to_split_on = [
    #     ("#", "Heading 1"),
    #     ("##", "Heading 2"),
    #     ("###", "Heading 3"),
    #     ("####", "Heading 4"),
    #     ("\\n", "Paragraph"),
    #     ("\\n\\n", "Paragraph"),
    # ]
    # md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, return_each_line=False)
    # docs = []
    # for d in md_docs:
    #     split_docs = md_splitter.split_text(d.page_content)
    #     docs += [Document(page_content=doc.page_content, metadata=d.metadata) for doc in split_docs]
    return md_docs

def load_docs_from_dir(directory: str) -> list[Document]:
    docs = []
    print("Loading json documents...")
    docs = DirectoryLoader(path=directory, glob="**/*.json", recursive=True, show_progress=True, silent_errors=True, loader_cls=JSONLoader, loader_kwargs={"jq_schema": ".", "text_content": False}).load()
    print("Loading csv documents...")
    docs += DirectoryLoader(path=directory, glob="**/*.csv", recursive=True, show_progress=True, silent_errors=True, loader_cls=CSVLoader, loader_kwargs={"csv_args":{"delimiter": "\n"}}).load()
    print("Loading md documents...")
    docs += _load_md_docs(directory)
    print("Loading all remaining non-json/csv/md documents...")
    # load all non-json/csv/md files as text
    docs += DirectoryLoader(path=directory, glob="**/*[!(.json|.csv|.md)]", recursive=True, show_progress=True, silent_errors=True, loader_kwargs={"autodetect_encoding": True, "mode": "elements"}).load()
    return _split_documents(docs)
