# from langchain_anthropic import ChatAnthropic
# import os
# from dotenv import load_dotenv

# load_dotenv()

# print(os.getenv("ANTHROPIC_API_KEY"))

# model = ChatAnthropic(model = "claude-sonnet-5-5")

# result = model.invoke("What is the capital of India?")

# print(result)



from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
import os

load_dotenv()

api_key = os.getenv("ANTHROPIC_API_KEY")
print("API key loaded:", bool(api_key))

model = ChatAnthropic(model="claude-sonnet-5-5")
print(model.invoke("What is the capital of India?").content)