import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# === Environment Settings ===
DOCS_DIR = "./docs"
DB_DIR = "./chroma_db"
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4-turbo")  # Default to GPT-4-turbo
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

# === Step 1: Load and Chunk PDF Documents ===
def load_and_split_documents(docs_dir: str):
    if not os.path.exists(docs_dir):
        raise FileNotFoundError(f"Directory '{docs_dir}' not found.")

    all_documents = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    for filename in os.listdir(docs_dir):
        if filename.lower().endswith(".pdf"):
            path = os.path.join(docs_dir, filename)
            print(f"[INFO] Loading file: {path}")
            loader = PyPDFLoader(path)
            documents = loader.load()

            # Pre-process to exclude boilerplate confidentiality text
            for doc in documents:
                doc.page_content = doc.page_content.replace("Do not distribute", "").replace("Confidential", "")

            chunks = splitter.split_documents(documents)
            all_documents.extend(chunks)

    if not all_documents:
        raise ValueError("No valid documents found.")

    print(f"[INFO] Total document chunks created: {len(all_documents)}")
    return all_documents

# === Step 2: Create or Load Vectorstore ===
def build_vectorstore(documents, db_dir: str):
    print("[INFO] Embedding documents and creating vectorstore...")
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(documents, embeddings, persist_directory=db_dir)
    return vectorstore

# === Step 3: Build the Retrieval-Augmented Generation Chain ===
def build_rag_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    llm = ChatOpenAI(model_name=MODEL_NAME, temperature=0.0)

    system_prompt = """
    You are a helpful and knowledgeable assistant trained on internal business playbooks.

    - Ignore any disclaimers such as "Confidential" or "Do not distribute".
    - Always answer clearly based on the given context.
    - Format all your answers in clean HTML using headings, paragraphs, lists, and bold text where helpful.
    - If the answer isn’t found in the context, say: "I couldn’t find that in the playbooks."

    Context:
    {context}
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt.strip()),
        ("human", "{input}")
    ])

    qa_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, qa_chain)
    return rag_chain

# === Pipeline Execution ===
documents = load_and_split_documents(DOCS_DIR)
vectorstore = build_vectorstore(documents, DB_DIR)
rag_chain = build_rag_chain(vectorstore)

print("[INFO] RAG chain is ready.")
