from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_classic.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.prompts import PromptTemplate

load_dotenv()


model = ChatGoogleGenerativeAI(model = "gemini-3.5-flash-lite")


schema = [
    ResponseSchema(name="fact_1", description="Fact 1 about topic"),
    ResponseSchema(name="fact_2", description="Fact 2 about topic"),
    ResponseSchema(name="fact_3", description="Fact 3 about topic"),
]


parser = StructuredOutputParser.from_response_schemas(schema)


template = PromptTemplate (
    template = "Give me 3 facts about {topic} \n {format_instruction}",
    input_variables=["topic"],
    partial_variables={"format_instruction": parser.get_format_instructions()}
)


prompt = template.invoke({"topic": "black hole"})

result = model.invoke(prompt)
content = result.content

if isinstance(content, list):
    content = "".join(
        block["text"]
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    )

final_result = parser.parse(content)

print(final_result)

# final_result = parser.parse(result.content)

# print(final_result)

# chain = template | model | parser

# result = chain.invoke({"topic": "Virat Kohli"})

# print(result)

