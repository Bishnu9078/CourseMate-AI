# Copyright (c) 2026 Bishnuprasad Sahoo (GitHub: Bishnu9078)
# Licensed under the MIT License. See LICENSE in the project root.

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings


docs = [
    Document(page_content="Gradient descent is an optimization algorithm used in machine learning."),
    Document(page_content="Gradient descent minimizes the loss function."),
    Document(page_content="Gradient descent is an optimization that minimizes the loss function."),
    Document(page_content="Neural networks use gradient descent for training."),
    Document(page_content="Support Vector Machines are supervised learning algorithms."),
]

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = InMemoryVectorStore.from_documents(docs, embeddings)

similarity_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)

print("\n===== Similarity Search Results =====\n")
for doc in similarity_retriever.invoke("What is gradient descent?"):
    print(doc.page_content)

mmr_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "fetch_k": 5, "lambda_mult": 0.5},
)

print("\n===== MMR Results =====\n")
for doc in mmr_retriever.invoke("What is gradient descent?"):
    print(doc.page_content)
