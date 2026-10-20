from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field


load_dotenv()


model = ChatGoogleGenerativeAI(model = "gemini-3.5-flash-lite")

class Person(BaseModel):
    name: str = Field(description="Name of the person")
    age: int = Field(description="Age of the person")
    city: str = Field(description="Name of the city ;of the person belongs to")

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate (
    template = "Generate the name, age and city of a fictional {place} person \n {format_instruction} ",
    input_variables=["place"],
    partial_variables={"format_instruction": parser.get_format_instructions()}
)

# prompt = template.invoke({"place": "Uttar Pradesh"})

# result = model.invoke(prompt)
# content = result.text if hasattr(result, "text") else result.content

# if isinstance(content, list):
#     content = "".join(
#         block["text"]
#         for block in content
#         if isinstance(block, dict) and block.get("type") == "text"
#     )

# final_result = parser.parse(content)
# print(final_result)

def extract_text(message):
    return message.text()

chain = template | model | RunnableLambda(extract_text) | parser

result = chain.invoke({"place": "Uttar Pradesh"})

print(result)
print(type(result))
print(result.name)
print(result.age)
print(result.city)