import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore # Updated import
from langchain_openai import OpenAIEmbeddings
from medicalai.helper import document_loader, split_documents

load_dotenv()

# Setup API Keys
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Initialize Pinecone Client
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "medicalrag"

# 1. Check and Create Index
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536, 
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# 2. Process Documents
data_path = "Data" 
document = document_loader(path=data_path)
chunks = split_documents(documents=document)

# 3. Initialize Embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 4. Load documents into Pinecone via LangChain
# Note: Use index_name (string) here
vectorstore = PineconeVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    index_name=index_name
)

print("Ingestion complete. Documents are now searchable in Pinecone.")