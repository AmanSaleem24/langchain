from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
from functools import lru_cache

from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from youtube_transcript_api import YouTubeTranscriptApi

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["chrome-extension://YOUR_EXTENSION_ID"],
    allow_methods=["POST"],
    allow_headers=["Content-Type"],
)

DATA_ROOT = Path("data")
DATA_ROOT.mkdir(exist_ok=True)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

llm = ChatOllama(model="llama3.2:3b", temperature=0)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

class VideoRequest(BaseModel):
    video_id: str

class QuestionRequest(BaseModel):
    video_id: str
    question: str

def get_transcript(video_id: str) -> str:
    folder = DATA_ROOT / video_id
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "transcript.txt"

    if path.exists():
        return path.read_text(encoding="utf-8")

    fetched = YouTubeTranscriptApi().fetch(
        video_id, languages=["en"]
    )
    transcript = " ".join(s.text for s in fetched)
    path.write_text(transcript, encoding="utf-8")
    return transcript

@lru_cache(maxsize=8)
def get_vector_store(video_id: str):
    folder = DATA_ROOT / video_id
    index_path = folder / "faiss_index"

    transcript = get_transcript(video_id)

    if (index_path / "index.faiss").exists() and (
        index_path / "index.pkl"
    ).exists():
        return FAISS.load_local(
            str(index_path),
            embeddings,
            allow_dangerous_deserialization=True,
        )

    chunks = splitter.create_documents([transcript])
    store = FAISS.from_documents(chunks, embeddings)
    store.save_local(str(index_path))
    return store

@app.post("/ask")
def ask(req: QuestionRequest):
    try:
        store = get_vector_store(req.video_id)
        docs = store.similarity_search(req.question, k=4)
        context = "\n\n".join(d.page_content for d in docs)

        response = llm.invoke(
            "Answer only using the video transcript context. "
            "If it doesn't contain the answer, say you don't know.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {req.question}"
        )
        return {"answer": response.content}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@app.post("/summarize")
def summarize(req: VideoRequest):
    try:
        transcript = get_transcript(req.video_id)
        chunks = splitter.create_documents([transcript])

        # Summarize in batches to limit context size.
        summaries = []
        for chunk in chunks:
            result = llm.invoke(
                "Summarize this transcript excerpt, preserving "
                "important facts and topics:\n\n"
                + chunk.page_content
            )
            summaries.append(result.content)

        # Hierarchical reduction for long videos.
        while len(summaries) > 1:
            reduced = []
            for i in range(0, len(summaries), 4):
                batch = "\n\n".join(summaries[i:i + 4])
                result = llm.invoke(
                    "Combine these excerpt summaries into a "
                    "concise, coherent summary. Preserve key "
                    "topics and avoid repetition:\n\n" + batch
                )
                reduced.append(result.content)
            summaries = reduced

        return {"summary": summaries[0] if summaries else ""}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
