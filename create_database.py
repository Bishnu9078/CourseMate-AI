from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

# PDF file path
pdf_path = Path(__file__).parent / "document loaders" / "deeplearning.pdf"

# Load PDF
loader = PyPDFLoader(str(pdf_path))
docs = loader.load()

print("Pages loaded:", len(docs))

# Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

chunks = splitter.split_documents(docs)
print("Chunks created:", len(chunks))

# Create embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Store chunks in ChromaDB
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    collection_name="deep_learning",
    persist_directory=str(Path(__file__).parent / "chroma_db"),
)

print("PDF successfully stored in ChromaDB!")
