from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model = "gemini-3.5-flash-lite")

prompt = PromptTemplate(
    template = "Write a summary for the following poem - \n {poem}",
    input_variables=["poem"]
)
parser = StrOutputParser()

loader = TextLoader('cricket.txt', encoding='utf-8')

docs = loader.load()

doc = docs[0]
print(doc.metadata)

chain = prompt | model | parser

result = chain.invoke({'poem': doc.page_content})

print(result)