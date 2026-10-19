from langchain_anthropic import ChatAnthropic
import os
from dotenv import load_dotenv

load_dotenv()

print(os.getenv("OPENAI_API_KEY"))

model = ChatAnthropic(model = "gpt-4", temperature=0, max_completion_tokens=
                   50)

result = model.invoke("What is the capital of India?")



