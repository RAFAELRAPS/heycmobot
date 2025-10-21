# chatbot.py
from langchain_core.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import os

# Load and split PDF documents
docs_dir = "./docs"
all_documents = []
for filename in os.listdir(docs_dir):
    if filename.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(docs_dir, filename))
        pdf_documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(pdf_documents)
        all_documents.extend(chunks)

# Embed and create vector store
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(all_documents, embeddings, persist_directory="./chroma_db")
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# LLM and QA chain
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.0)
system_prompt = """
Use the given context to answer the user's question.
If you don't know the answer, say you don't know.
Keep the answer concise (max three sentences).
Context: {context}
"""
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])
qa_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, qa_chain)
