# app.py
from fastapi import FastAPI, Request
from pydantic import BaseModel
from chatbot import rag_chain

app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/ask")
async def ask(query: Query):
    result = rag_chain.invoke({"input": query.question})
    return {
        "answer": result["answer"],
        "sources": [doc.metadata.get("source", "") for doc in result.get("context", [])]
    }
