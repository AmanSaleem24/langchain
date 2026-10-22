from pathlib import Path
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# Configuration
# VIDEO_ID = "Gfr50f6ZBvo"
VIDEO_ID = "SZ1sDsmzs6s"
DATA_DIR = Path("data") / VIDEO_ID
TRANSCRIPT_PATH = DATA_DIR / "transcript.txt"
INDEX_PATH = DATA_DIR / "faiss_index"

def format_docs(retrieved_docs):
    context_text = "\n\n.".join(doc.page_content for doc in retrieved_docs)
    return context_text

DATA_DIR.mkdir(parents=True, exist_ok=True)

# Local embedding and LLM models
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

parser = StrOutputParser()

# Step 1: Fetch transcript only if it isn't saved locally
if TRANSCRIPT_PATH.exists():
    transcript = TRANSCRIPT_PATH.read_text(encoding="utf-8")
    print("Loaded transcript from local storage.")
else:
    print("Fetching transcript from YouTube...")

    ytt_api = YouTubeTranscriptApi()
    fetched_transcript = ytt_api.fetch(
        VIDEO_ID,
        languages=["en"]
    )

    transcript = " ".join(
        snippet.text for snippet in fetched_transcript
    )

    TRANSCRIPT_PATH.write_text(transcript, encoding="utf-8")
    print("Transcript saved locally.")

print("Transcript length:", len(transcript))

# Step 2: Load FAISS if it exists; otherwise chunk and embed
if (INDEX_PATH / "index.faiss").exists() and (
    INDEX_PATH / "index.pkl").exists():
        vector_store = FAISS.load_local(
            str(INDEX_PATH),
            embedding_model,
            allow_dangerous_deserialization=True
        )
        print("Loaded existing FAISS index. Skipping chunking and embedding.")

else:
    print("Creating chunks...")
    chunks = splitter.create_documents([transcript])
    print("Number of chunks:", len(chunks))

    print("Generating embeddings and creating FAISS index...")
    vector_store = FAISS.from_documents(
        chunks,
        embedding_model
    )

    vector_store.save_local(str(INDEX_PATH))
    print("FAISS index saved locally.")

# Step 3: Retrieve relevant chunks
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)


prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables = ['context', 'question']
)


parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})

main_chain = parallel_chain | prompt | llm | parser

result = main_chain.invoke("Summarise the video")

print(result)
