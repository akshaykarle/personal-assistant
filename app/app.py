from langchain.document_loaders import WebBaseLoader
from langchain.embeddings import GPT4AllEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain import hub
from langchain.chains import RetrievalQA

loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
data = loader.load()

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

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

question = "What are the approaches to Task Decomposition?"
qa_chain({"query": question})
