from operator import itemgetter

from langchain_core.messages import AIMessage
from langchain.schema.runnable import Runnable, RunnableParallel, RunnableLambda
from langchain.llms import LlamaCpp
from langchain.memory import ConversationBufferMemory
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)

from db import Database
from ingest import load_docs_from_dir


REDIS_URL = "redis://localhost:6379/0"


def configure_retriever(db: Database):
    # Read documents
    if db.is_empty():
        print("Database is empty, loading documents...")
        docs = load_docs_from_dir("../data")
        db.store_documents(docs)
    else:
        print("Database is ready, preparing retriever...")

    retriever = db.docstore.as_retriever(search_type="similarity", search_kwargs={"k": 8, "fetch_k": 100, "lambda_mult": 0.50})

    return retriever


def setup_chain(memory: ConversationBufferMemory) -> Runnable:
    db = Database(collection_name="personal_assistant")
    retriever = configure_retriever(db)

    # Setup LLM and QA chain
    llm = LlamaCpp(
        model_path="../models/llama-cpp/llama-2-13b-chat.Q4_K_M.gguf",
        temperature=0.5,
        max_tokens=3000,
        top_p=1,
        verbose=True,  # Verbose is required to pass to the callback manager
        n_ctx=3000,
        streaming=True,
    )

    # Prompt
    template = """
    <<SYS>>You are a helpful, respectful and honest chatbot assistant for question-answering tasks. Use the following pieces of retrieved context and chat history to answer questions and have a conversation.
    Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
    Answer in two to three sentences.<</SYS>>

    Context: {context}
    """
    prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(template),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("[INST]{question}[/INST]"),
            AIMessage(content="Answer: "),
        ])

    setup_and_retrieval = RunnableParallel({
        "context": itemgetter("question") | retriever,
        "chat_history": RunnableLambda(memory.load_memory_variables) | itemgetter("chat_history"),
        "question": itemgetter("question"),
    })

    chain = setup_and_retrieval | prompt | llm | StrOutputParser
    return chain
