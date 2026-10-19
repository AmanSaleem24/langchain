# from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

# llm = HuggingFacePipeline.from_model_id(
#     model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
#     task="text-generation",
#     pipeline_kwargs=dict(
#         temperature = 0.7,
#     )
# )
# model = ChatHuggingFace(llm = llm)

# while True:
#     user_input = input("You: ")
#     if user_input == "exit":
#         break
#     result = model.invoke(user_input)
#     print("AI: ", result.content)


from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_id)
base_model = AutoModelForCausalLM.from_pretrained(model_id)

pipe = pipeline(
    "text-generation",
    model=base_model,
    tokenizer=tokenizer,
    max_new_tokens=128,
    do_sample=True,
    return_full_text=False,
)

llm = HuggingFacePipeline(pipeline=pipe)
model = ChatHuggingFace(llm=llm)

while True:
    user_input = input("You: ")

    if user_input.strip().lower() == "exit":
        break

    result = model.invoke(user_input)
    print("AI:", result.content)