from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task = "text-generation",
    pipeline_kwargs=dict(
        temperature=0.2,
    )
)
model = ChatHuggingFace(llm = llm)


template1 = PromptTemplate(
    template = "Write a report on the following topic {topic}",
    input_variables=["topic"]
)

template2 = PromptTemplate(
    template = "Write a 100 words summary on the following text.\n {text}", 
    input_variables=["text"]
)

prompt1 = template1.invoke({"topic": "Black Hole"})

result = model.invoke(prompt1)

print("\n\nResult of prompt 1", result.content)

prompt2 = template2.invoke({"text": result.content})

result2 = model.invoke(prompt2)

print("Result of prompt 2 ",result2.content)



