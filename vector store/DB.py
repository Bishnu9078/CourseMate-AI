from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings


docs = [
    Document(
        page_content="Python is widely used in Artificial Intelligence.",
        metadata={"source": "AI_book"},
    ),
    Document(
        page_content="Pandas is used for data analysis in Python.",
        metadata={"source": "DataScience_book"},
    ),
    Document(
        page_content="Neural networks are used in deep learning.",
        metadata={"source": "DL_book"},
    ),
]

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)
vectorstore = InMemoryVectorStore.from_documents(docs, embedding_model)

print("Documents loaded into the in-memory vector store.")
for doc in vectorstore.similarity_search("What is used for data analysis?", k=2):
    print("\nContent:", doc.page_content)
    print("Metadata:", doc.metadata)

retriever = vectorstore.as_retriever()
for doc in retriever.invoke("Explain deep learning"):
    print(doc.page_content)
