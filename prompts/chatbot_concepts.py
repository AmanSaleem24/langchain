"""
Step-by-step chatbot concepts: messages, placeholders, templates, and a prompt generator.

Run:  pip install langchain langchain-openai  (or any LLM framework)
This example uses LangChain-style classes. The same *ideas* apply in most frameworks.
"""

from langchain_core.prompts import PromptTemplate
from langchain_core.messages import (
    SystemMessage, HumanMessage, AIMessage,
)


# ---------------------------------------------------------------------------
# 1) MESSAGES  (the building blocks of a conversation)
# ---------------------------------------------------------------------------
# An LLM chat is just a list of messages, in order:
#   SystemMessage  -> the rules / personality / role for the bot
#   HumanMessage   -> what the user said
#   AIMessage      -> what the bot replied
#   (and it can alternate: user, bot, user, bot, ...)

system = SystemMessage(
    content="You are a friendly customer-support bot for 'Acme Shoes'."
)

user_msg = HumanMessage(content="Do you ship internationally?")

previous_bot_reply = AIMessage(content="Yes! We ship to most countries.")

# A 'conversation' is just this list, kept in order:
conversation = [system, user_msg, previous_bot_reply]

print("conversation:", conversation)


# ---------------------------------------------------------------------------
# 2) MESSAGE_PLACEHOLDER  (a gap in a template you fill later)
# ---------------------------------------------------------------------------
# Instead of hard-coding the user's text into the prompt, we leave a named
# gap. That gap is a "placeholder". Later we plug a real value into it.

template_with_one_gap = PromptTemplate(
    template="Answer this question about Acme Shoes: {user_question}",
    input_variables=["user_question"],
)

# "user_question" is a MESSAGE_PLACEHOLDER in disguise: a named slot that
# gets filled with the user's actual message when we invoke the template.
print(template_with_one_gap.invoke({"user_question": "What colors do you have?"}))


# ---------------------------------------------------------------------------
# 3) CHAT_PROMPT_TEMPLATE  (a structured, multi-role template)
# ---------------------------------------------------------------------------
# A ChatPromptTemplate lets you build a *list* of messages with placeholders
# in different roles (system, human, ai). This is the most common pattern.

from langchain_core.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful bot. Keep answers under 50 words."),
    ("human", "{user_input}"),
])

# See: the {user_input} placeholder lives inside a *human* message.
chat_prompt = chat_template.invoke({"user_input": "Do you have red shoes?"})
print("\nchat_prompt (a message list):", chat_prompt)

# `chat_prompt` is what you now feed to the LLM:
# from langchain_openai import ChatOpenAI
# llm = ChatOpenAI()
# answer = llm.invoke(chat_prompt)


# ---------------------------------------------------------------------------
# 4) PROMPT_GENERATOR  (a reusable function that BUILDS the final prompt)
# ---------------------------------------------------------------------------
# In real apps the prompt depends on several dynamic inputs:
#   - the user's new question
#   - the conversation history (so the bot has memory)
#   - retrieved context (docs/knowledge)
# A "prompt generator" is just a reusable piece of code that takes those
# raw inputs and returns a *ready-to-send* prompt (a list of messages).

def generate_chat_prompt(user_question: str, history: list, context: str = ""):
    """
    This is our prompt generator.

    user_question : the brand-new user message (a HumanMessage or string)
    history       : past [system/human/ai] messages (gives the bot memory)
    context       : optional extra knowledge to include
    """
    messages = []

    # 1) base system instructions
    system_text = "You are Acme Shoes support. Be friendly and concise."
    if context:
        system_text += f"\n\nRelevant info:\n{context}"
    messages.append(SystemMessage(content=system_text))

    # 2) remembered history
    messages.extend(history)

    # 3) the new user turn
    messages.append(HumanMessage(content=user_question))

    return messages


# Example of using the generator:
history = [
    HumanMessage(content="Hi, I need help."),
    AIMessage(content="Sure, what do you need?"),
]

final_prompt = generate_chat_prompt(
    user_question="Do you ship to Germany?",
    history=history,
    context="We ship to the EU, including Germany.",
)
print("\nFinal generated prompt (ready to send to the LLM):")
for m in final_prompt:
    print(f"  [{m.type}] {m.content}")
