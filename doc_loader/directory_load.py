from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path="./pdfs",
    glob='*.pdf',
    loader_cls=PyPDFLoader
)

docs = loader.load()

print(len(docs))
print(docs[110].page_content)
print(docs[110].metadata)