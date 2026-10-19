from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

load_dotenv()

print(os.getenv("OPENAI_API_KEY"))

model = ChatGoogleGenerativeAI(model = "gemini-3.5-flash-lite")

result = model.invoke("What is the capital of India?")

print(result)


