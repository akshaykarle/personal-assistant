from db import Database
from ingest import load_docs_from_dir
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

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

retriever = db.docstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

prompt_template = """
You are an assistant working for Sahaj Software Consultancy for specialized for question-answering tasks. Use the following pieces of retrieved context from the Sahaj database to answer the question. If you don't know the answer, just say that you don't know. Use five sentences maximum and keep the answer concise.
Question: {question}
Context: {context}
Answer:"""

rag_prompt = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    # chain_type="map_reduce",
    return_source_documents=True,
    retriever=retriever,
    chain_type_kwargs={"prompt": rag_prompt},
)

query = "In which projects did we use ML pipelines?"
query2 = "Have we done any work in the financial industry? If so, which client did we work for?"
query3 = "What is the status of infrastructure in Canada in Talon Ada project?"
relevant_docs = retriever.get_relevant_documents(query3)
print(len(relevant_docs))
print(relevant_docs)
response = qa(query3)
print(f"""
        Question: {response['query']}
        Answer: {response['result']}

        No. of Source documents: {len(response['source_documents'])}
        Source documents: {response['source_documents']}
""")
