from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load all PDFs from 'docs' folder
docs_path = "docs"
all_pages = []

for filename in os.listdir(docs_path):
    if filename.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(docs_path, filename))
        pages = loader.load_and_split()
        all_pages.extend(pages)

# Split documents into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
chunks = splitter.split_documents(all_pages)

# Embed using OpenAI
embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")
vectorstore = Chroma.from_documents(chunks, embedding_model, persist_directory="db")
vectorstore.persist()

# Create QA chain using GPT-4
llm = ChatOpenAI(model="gpt-4", temperature=0.2)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)
