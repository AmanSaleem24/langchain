from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = GoogleGenerativeAI(
        model = "gemini-3.5-flash-lite",
        temperature=0.7,
        max_output_tokens=50,
        timeout=30,
        max_retries=2,
    )

chat_history = []

while True:
    user_input = input("You: ")
    chat_history.append(user_input)
    if user_input.strip().lower() == "exit":
        break
    result = model.invoke(chat_history)
    chat_history.append(result)
    print("AI: ", result)

print(chat_history)
