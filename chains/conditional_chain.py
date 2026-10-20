from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from langchain_classic.schema.runnable import RunnableBranch, RunnableLambda, RunnableParallel

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

parser1 = StrOutputParser()

class Feedback(BaseModel):
    sentiment : Literal["positive", "negative"] = Field(description="Give the sentiment of the feedback")

parser2 = PydanticOutputParser(pydantic_object=Feedback)


prompt1 = PromptTemplate(
    template = "Classify the sentiment of the following feedback text into positive or negative\n Text - > {feedback} \n {format_instruction}",
    input_variables=["feedback"],
    partial_variables={"format_instruction": parser2.get_format_instructions()}
)

classifier_chain = prompt1 | model | parser2

parallel_chain = RunnableParallel({
    "feedback": lambda x: x["feedback"],
    "classification": classifier_chain
})

prompt2 = PromptTemplate(
    template = "Write an appropriate response to this positive feedback \n {feedback}",
    input_variables=["feedback"]
)


prompt3 = PromptTemplate(
    template = "Write an appropriate response to this negative feedback \n {feedback}",
    input_variables=["feedback"]
)


branch_chain = RunnableBranch(
    (
        lambda x: x["classification"].sentiment == "positive",
        RunnableLambda(lambda x: x["feedback"]) | prompt2 | model | parser1
    ),
    (
        lambda x: x["classification"].sentiment == "negative",
        RunnableLambda(lambda x: x["feedback"]) | prompt3 | model | parser1
    ),
    RunnableLambda(lambda x: "Could not find sentiment")
)

chain = parallel_chain | branch_chain

result = chain.invoke({'feedback': "This smartphone has terrible battery life, a laggy interface, and a disappointing camera. The phone frequently freezes, and the overall performance is nowhere near what I expected for the price. I regret buying it."})

print(result)

chain.get_graph().print_ascii()
