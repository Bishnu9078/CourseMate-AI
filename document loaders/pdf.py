from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


with open("state ") as f:
    state_of_the_union = f.read()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)

data = PyPDFLoader("document loaders/GRU.pdf")
docs = data.load()

splitter = TokenTextSplitter(
    chunk_size=200,
    chunk_overlap=10
)


chunks = splitter.split_documents(docs)
print(chunks[0].page_content)