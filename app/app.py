from db import Database
from ingest import load_docs_from_dir
from langchain.llms import GPT4All
from langchain import hub
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain

db = Database()
if db.is_empty():
    docs = load_docs_from_dir("../data")
    db.store_documents(docs)

query = "What category is Transport of London in?"

llm = GPT4All(
    model="../models/gpt4all/gpt4all-falcon-q4_0.gguf",
)

rag_prompt = hub.pull("smithing-gold/inline-citations")

qa = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm,
    chain_type="map_reduce",
    return_source_documents=True,
    retriever=db.as_retriever(),
)

response = qa(query)
