from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()

# llm = HuggingFacePipeline.from_model_id(
#     model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
#     task = "text-generation",
#     pipeline_kwargs=dict(
#         temperature=0.5,
#         max_new_tokens=200,
#     )
# )

model = ChatGoogleGenerativeAI(model = "gemini-3.5-flash-lite")
# model = ChatHuggingFace(llm =llm)

parser = JsonOutputParser()

template = PromptTemplate (
    template = "Give me 5 facts about {topic} \n {format_instruction}",
    input_variables=["topic"],
    partial_variables={"format_instruction": parser.get_format_instructions()}
)

# prompt = template.format()

# print(prompt)

# result = model.invoke(prompt)

# print(result.content)

# ans = parser.parse(result.content)

# print("FInal result", ans)

# print(type(ans))


chain = template | model | parser

result = chain.invoke({"topic": "Virat Kohli"})

print(result)

