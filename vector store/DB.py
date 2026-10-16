
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Sample documents
docs = [
    Document(
        page_content="Python is widely used in Artificial Intelligence.",
        metadata={"source": "AI_book"}
    ),
    Document(
        page_content="Pandas is used for data analysis in Python.",
        metadata={"source": "DataScience_book"}
    ),
    Document(
        page_content="Neural networks are used in deep learning.",
        metadata={"source": "DL_book"}
    ),
]

# Create embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Store documents in ChromaDB
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    collection_name="rag_documents",
    persist_directory="chroma_db"
)

print("Documents stored successfully in ChromaDB!")

# Test retrieval
results = vectorstore.similarity_search(
    "What is used for data analysis?",
    k=2
)

for doc in results:
    print("\nContent:", doc.page_content)
    print("Metadata:", doc.metadata)


retriever = vectorstore.as_retriever()

docs = retriever.invoke("Explain deep learning")

for d in docs:
    print(d.page_content)