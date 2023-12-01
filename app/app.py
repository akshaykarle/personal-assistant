from db import Database
from ingest import load_docs_from_dir
from langchain.llms import LlamaCpp
from langchain import hub
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain

db = Database(collection_name="personal_assistant")
if db.is_empty():
    print("Database is empty, loading documents...")
    docs = load_docs_from_dir("../data")
    db.store_documents(docs)


llm = LlamaCpp(
    model_path="../models/llama-cpp/llama-2-13b-chat.gguf.q4_K_S.bin",
    temperature=0.75,
    max_tokens=3000,
    top_p=1,
    verbose=True,  # Verbose is required to pass to the callback manager
    n_ctx=3000,
)

retriever = db.as_retriever()

qa = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm,
    chain_type="map_reduce",
    return_source_documents=True,
    retriever=retriever,
)

query = "In which projects did we use ML pipelines?"
relevant_docs = retriever.get_relevant_documents(query)
print(relevant_docs)
response = qa(query)
print(f"""
        Question: {response['question']}
        Answer: {response['answer']}
        Sources: {response['sources']}

        Source documents: {response['source_documents']}
""")
