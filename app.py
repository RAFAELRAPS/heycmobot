from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from chatbot import rag_chain

app = FastAPI()

# ✅ Enable CORS for your WordPress frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your domain for production, e.g., ["https://yourwordpresssite.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Request body schema
class Query(BaseModel):
    question: str

# ✅ Make sure your frontend POSTs to this exact path: /chat
@app.post("/chat")
async def chat(query: Query):
    result = rag_chain.invoke({"input": query.question})

    return {
        "answer": result.get("answer", "No answer returned."),
        "sources": [doc.metadata.get("source", "") for doc in result.get("context", [])]
    }
