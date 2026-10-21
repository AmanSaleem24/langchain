from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('LostLink_Review_Paper.pdf')

docs = loader.load()


splitter = CharacterTextSplitter(
    chunk_size = 100, 
    chunk_overlap = 10,
    separator= ""
)
result = splitter.split_documents(docs)

print(len(result))
# print(result[0])