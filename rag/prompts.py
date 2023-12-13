from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate
)

# Used to condense a question and chat history into a single question
condense_question_prompt_template = """<<SYS>>Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language. If there is no chat history, just rephrase the question to be a standalone question.<</SYS>>

Chat History:
{chat_history}
Follow Up Input: [INST]{question}[/INST]
"""  # noqa: E501
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(
    condense_question_prompt_template
)

# RAG Prompt to provide the context and question for LLM to answer
# We also ask the LLM to cite the source of the passage it is answering from
llm_context_prompt_template = """
<<SYS>>You are a helpful, respectful and honest chatbot assistant for question-answering tasks.
Use the following pieces of retrieved context to answer questions and have a conversation.
Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
Answer in two to three sentences.<</SYS>>

----
{context}
----
Question: [INST]{question}[/INST]
"""  # noqa: E501

LLM_CONTEXT_PROMPT = ChatPromptTemplate.from_template(llm_context_prompt_template)

# Used to build a context window from passages retrieved
document_prompt_template = """
---
NAME: {source}
PASSAGE:
{page_content}
---
"""

DOCUMENT_PROMPT = PromptTemplate.from_template(document_prompt_template)
