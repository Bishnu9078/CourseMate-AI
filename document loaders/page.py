from langchain_community.document_loaders import WebBaseLoader

url = "https://www.apple.com/in/shop/buy-mac/mac-mini/m6-chip-12-core-cpu-12-core-gpu-16gb-memory-256gb-storage"

data = WebBaseLoader(url)

docs = data.load()

print(docs[0].page_content)