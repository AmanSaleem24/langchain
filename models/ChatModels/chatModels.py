from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

print(os.getenv("OPENAI_API_KEY"))

model = ChatOpenAI(model = "gpt-4", temperature=0, max_completion_tokens=
                   50)

result = model.invoke("What is the capital of India?")



