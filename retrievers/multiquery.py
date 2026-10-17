# Copyright (c) 2026 Bishnuprasad Sahoo (GitHub: Bishnu9078)
# Licensed under the MIT License. See LICENSE in the project root.

from dotenv import load_dotenv
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings


load_dotenv()

docs = [
    Document(page_content="Gradient descent is an optimization algorithm used in machine learning."),
    Document(page_content="Gradient descent minimizes the loss function."),
    Document(page_content="Gradient descent is an optimization that minimizes the loss function."),
    Document(page_content="Neural networks use gradient descent for training."),
    Document(page_content="Support Vector Machines are supervised learning algorithms."),
]

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = InMemoryVectorStore.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
multi_query_retriever = MultiQueryRetriever.from_llm(retriever=retriever, llm=llm)

retrieved_docs = multi_query_retriever.invoke("What is gradient descent?")
print("\nRetrieved Documents:\n")
for doc in retrieved_docs:
    print(f"- {doc.page_content}")
