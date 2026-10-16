
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

# Embedding model (must match the model used to create chroma_db)
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load existing ChromaDB
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model,
    collection_name="deep_learning"
)

# Create retriever using MMR
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 10,
        "lambda_mult": 0.5
    }
)

# Initialize Groq LLM
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.
If the answer is not present in the context, say:
"I could not find the answer in the document."
"""
    ),
    (
        "human",
        """Context:
{context}

Question:
{question}
"""
    )
])

print("RAG system created successfully!")
print("Press 0 to exit.")

while True:
    query = input("\nYou: ").strip()

    if query == "0":
        print("Goodbye!")
        break

    if not query:
        continue

    # Retrieve relevant documents
    docs = retriever.invoke(query)

    if not docs:
        print("\nAI: I could not find the answer in the document.")
        continue

    context = "\n\n".join(doc.page_content for doc in docs)

    # Generate answer using Groq
    final_prompt = prompt.invoke({
        "context": context,
        "question": query
    })

    response = llm.invoke(final_prompt)

    print(f"\nAI: {response.content}")
