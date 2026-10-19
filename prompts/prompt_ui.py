from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate, load_prompt
import streamlit as st


load_dotenv()


model = GoogleGenerativeAI(
        model = "gemini-3.5-flash-lite",
        temperature=0.7,
        max_output_tokens=500,
        timeout=30,
        max_retries=2,
    )


st.header("Research Tool")

paper_input = st.selectbox( "Select Research Paper Name", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"] )

style_input = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"] ) 

length_input = st.selectbox( "Select Explanation Length", ["Short (1/2 paragraphs)", "Medium (2 paragraphs)", "Long (detailed explanation)"] )

template = load_prompt("template.json")

if st.button("Summarize"):
    chain = template | model
    result = chain.invoke({
        'paper_input': paper_input,
        'style_input': style_input,
        'length_input': length_input
    })
    st.write(result)
