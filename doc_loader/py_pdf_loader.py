from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader

from dotenv import load_dotenv

load_dotenv()

loader = PyPDFLoader('LostLink_Review_Paper.pdf')

docs = loader.load()

print(docs[1].page_content)
print(docs[1].metadata)

# model = ChatGoogleGenerativeAI(model = "gemini-3.5-flash-lite")

# prompt = PromptTemplate(
#     template = "",
#     input_variables=["poem"]
# )
# parser = StrOutputParser()



# chain = prompt | model | parser

# result = chain.invoke({'poem': doc.page_content})

# print(result)