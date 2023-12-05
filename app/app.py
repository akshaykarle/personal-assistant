from pprint import pprint
import os
from db import Database
from ingest import load_docs_from_dir
from langchain.llms import LlamaCpp
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
import streamlit as st

st.set_page_config(page_title="Personal Assistant", page_icon="🕵️‍♂️")
st.title("🕵️ Personal Assistant: Personalised help with daily tasks")


# @st.cache_resource(ttl="1h")
def configure_retriever(db: Database):
    # Read documents
    if db.is_empty():
        print("Database is empty, loading documents...")
        docs = load_docs_from_dir("../data")
        db.store_documents(docs)
    else:
        print("Database is ready, preparing retriever...")

    retriever = db.docstore.as_retriever(search_type="similarity", search_kwargs={"k": 6, "fetch_k": 100, "lambda_mult": 0.50})

    return retriever


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""):
        self.container = container
        self.text = initial_text
        self.run_id_ignore_token = None

    def on_llm_start(self, serialized: dict, prompts: list, **kwargs):
        # Workaround to prevent showing the rephrased question as output
        if prompts[0].startswith("Human"):
            self.run_id_ignore_token = kwargs.get("run_id")

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if self.run_id_ignore_token == kwargs.get("run_id", False):
            return
        self.text += token
        self.container.markdown(self.text)


class PrintRetrievalHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.status = container.status("**Context Retrieval**")

    def on_retriever_start(self, serialized: dict, query: str, **kwargs):
        self.status.write(f"**Question:** {query}")
        self.status.update(label=f"**Context Retrieval:** {query}")

    def on_retriever_end(self, documents, **kwargs):
        for idx, doc in enumerate(documents):
            source = os.path.basename(doc.metadata["source"])
            self.status.write(f"**Document {idx} from {source}**")
            self.status.markdown(doc.page_content)
        self.status.update(state="complete")


db = Database(collection_name="personal_assistant")
retriever = configure_retriever(db)

# Setup memory for contextual conversation
msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=msgs, return_messages=True)

# Setup LLM and QA chain
llm = LlamaCpp(
    model_path="../models/llama-cpp/llama-2-13b.Q4_K_M.gguf",
    temperature=0.75,
    max_tokens=3000,
    top_p=1,
    verbose=True,  # Verbose is required to pass to the callback manager
    n_ctx=3000,
)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    verbose=True,
)

if len(msgs.messages) == 0 or st.sidebar.button("Clear message history"):
    msgs.clear()
    msgs.add_ai_message("How can I help you?")

avatars = {"human": "user", "ai": "assistant"}
for msg in msgs.messages:
    st.chat_message(avatars[msg.type]).write(msg.content)

if user_query := st.chat_input(placeholder="Ask me anything!"):
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        retrieval_handler = PrintRetrievalHandler(st.container())
        stream_handler = StreamHandler(st.empty())
        response = qa_chain.run(user_query, callbacks=[retrieval_handler, stream_handler])
