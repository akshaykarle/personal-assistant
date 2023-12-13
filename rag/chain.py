from operator import itemgetter
from typing import List, Optional, Tuple

from langchain.llms import LlamaCpp
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough
from langchain.schema import BaseMessage, format_document
from langchain_core.pydantic_v1 import BaseModel, Field

from .db import Database
from .ingest import load_docs_from_dir
from .prompts import CONDENSE_QUESTION_PROMPT, DOCUMENT_PROMPT, LLM_CONTEXT_PROMPT


db = Database(collection_name="personal_assistant")
llm = LlamaCpp(
    model_path="../models/llama-cpp/llama-2-13b-chat.Q4_K_M.gguf",
    temperature=0.5,
    max_tokens=3000,
    top_p=1,
    verbose=True,  # Verbose is required to pass to the callback manager
    n_ctx=3000,
    streaming=True,
)

def _configure_retriever(db: Database):
    # Read documents
    if db.is_empty():
        print("Database is empty, loading documents...")
        docs = load_docs_from_dir("../data")
        db.store_documents(docs)
    else:
        print("Database is ready, preparing retriever...")

    retriever = db.docstore.as_retriever(search_type="similarity", search_kwargs={"k": 8, "fetch_k": 100, "lambda_mult": 0.50})

    return retriever

def _combine_documents(
    docs, document_prompt=DOCUMENT_PROMPT, document_separator="\n\n"
):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)


def _format_chat_history(chat_history: List[Tuple]) -> str:
    buffer = ""
    for dialogue_turn in chat_history:
        human = "Human: " + dialogue_turn[0]
        ai = "Assistant: " + dialogue_turn[1]
        buffer += "\n" + "\n".join([human, ai])
    return buffer

class ChainInput(BaseModel):
    chat_history: Optional[List[BaseMessage]] = Field(
        description="Previous chat messages."
    )
    question: str = Field(..., description="The question to answer.")


_inputs = RunnableParallel(
    standalone_question=RunnablePassthrough.assign(
        chat_history=lambda x: _format_chat_history(x["chat_history"])
    )
    | CONDENSE_QUESTION_PROMPT
    | llm
)

_retriever = _configure_retriever(db)
_context = {
    "context": itemgetter("standalone_question") | _retriever | _combine_documents,
    "question": lambda x: x["standalone_question"],
}

chain = _inputs | _context | LLM_CONTEXT_PROMPT | llm

chain = chain.with_types(input_type=ChainInput)
