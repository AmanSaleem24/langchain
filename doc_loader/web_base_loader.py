from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model = "gemini-3.5-flash-lite")

prompt = PromptTemplate(
    template = "Answer the following question\n {question} from the following {text}",
    input_variables=["question", "text"]
)
parser = StrOutputParser()

url = "https://www.flipkart.com/samsung-galaxy-s25-5g-snapdragon-icyblue-256-gb/p/itmb4ce5781b8a3b?pid=MOBHQCH6HGPPR9JS&lid=LSTMOBHQCH6HGPPR9JSIOXPZ6&marketplace=FLIPKART&q=s25&store=tyy%2F4io&srno=s_1_4&otracker=search&otracker1=search&fm=organic&iid=d1105112-445c-4e89-b97d-25efe85b2407.MOBHQCH6HGPPR9JS.SEARCH&ppt=None&ppn=None&ssid=lllpv6lj8w0000001792573876250&qH=d25fb707067c6b50&ov_redirect=true"

loader = WebBaseLoader(url)

docs = loader.load()

chain = prompt | model | parser

result = chain.invoke({"question": "Which colors are available for this smartphone?", "text": docs[0].page_content})

print(result)