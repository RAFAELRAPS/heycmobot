import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# === Config ===
DOCS_DIR = "./docs"
DB_DIR = "./chroma_db"
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

# === Improved System Prompt ===
system_prompt = """
You are a helpful and knowledgeable assistant trained on internal business playbooks. Your role is to provide clear, concise answers using only the provided context from these documents.

Ignore any disclaimers such as "Confidential" or "Do not distribute" — they are not relevant to your task.

Use the following guidelines when answering:
- Focus on extracting marketing strategies, operations, sales playbooks, and consulting workflows.
- Summarize frameworks or processes clearly when present in the context.
- Do not guess. If the answer is not in the context, reply: "I couldn’t find that in the playbooks."
- Responses should be short and precise (ideally 2–4 sentences).
- When helpful, format key ideas as bullet points.

Context: {context}
"""

def load_and_split_documents(docs_dir: str):
    """Loads PDFs and splits them into chunks for embedding."""
    if not os.path.exists(docs_dir):
        raise FileNotFoundError(f"Directory '{docs_dir}' not found.")
    
    all_documents = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    for filename in os.listdir(docs_dir):
        if filename.lower().endswith(".pdf"):
            path = os.path.join(docs_dir, filename)
            print(f"[INFO] Loading file: {path}")
            loader = PyPDFLoader(path)
            documents = loader.load()
            chunks = splitter.split_documents(documents)
            all_documents.extend(chunks)

    if not all_documents:
        raise ValueError("No documents found or all PDFs are empty.")
    
    print(f"[INFO] Total document chunks created: {len(all_documents)}")
    return all_documents

def build_vectorstore(documents, db_dir: str):
    """Creates or loads a Chroma vectorstore from document embeddings."""
    print("[INFO] Embedding documents and creating vectorstore...")
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(documents, embeddings, persist_directory=db_dir)
    return vectorstore

def build_rag_chain(vectorstore):
    """Builds the retrieval-augmented generation (RAG) chain."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    llm = ChatOpenAI(model_name=MODEL_NAME, temperature=0.0)

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt.strip()),
        ("human", "{input}")
    ])

    qa_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, qa_chain)
    return rag_chain

# === Chain Initialization ===
documents = load_and_split_documents(DOCS_DIR)
vectorstore = build_vectorstore(documents, DB_DIR)
rag_chain = build_rag_chain(vectorstore)

print("[INFO] RAG chain is ready.")
