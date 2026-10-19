from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()

print(os.getenv("HF_TOKEN"))

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-4B-Instruct-2507", task="text_generation")

model = ChatHuggingFace(llm=llm)

result = model.invoke("Who is the prime minister of India?")

print(result)