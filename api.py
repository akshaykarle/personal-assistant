from fastapi import FastAPI
from langserve import add_routes
import uvicorn

from rag import chain

app = FastAPI()
add_routes(app, chain, path="/llm")

uvicorn.run(app, host="localhost", port=8000)
