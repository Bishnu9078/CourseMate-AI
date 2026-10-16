from langchain_community.document_loaders import TextLoader

from langchain_text_splitters import CharacterTextSplitter

splitter = CharacterTextSplitter(
    separator="",  
    chunk_size=10,
    chunk_overlap=1
)

# Load the text file using UTF-8 encoding
data = TextLoader("document loaders/notes.txt")

docs = data.load()

# Split the loaded documents into smaller chunks
chunks = splitter.split_documents(docs)

for i in chunks:
    print(i.page_content)
    print()
    print()
    print()