import wikipedia

wikipedia.set_user_agent(
    "AmanLearningProject/1.0 (contact:piouskareem@gmail.com)"
)

from langchain_community.retrievers import WikipediaRetriever

retriever = WikipediaRetriever(
    top_k_results=2,
    lang="en"
)

query = "geopolitical history of India and Pakistan"

docs = retriever.invoke(query)

for i, doc in enumerate(docs):
    print(f"\n--- Result {i + 1} ---")
    print(f"Content:\n{doc.page_content}")