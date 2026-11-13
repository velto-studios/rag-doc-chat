"""FastAPI app: ingest documents, ask questions, get cited answers."""
from __future__ import annotations

import pathlib
import uuid

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .chunking import chunk_text
from .llm import extractive_answer, generate_answer, generative_available
from .retriever import TfidfRetriever

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent

app = FastAPI(title="rag-doc-chat", version="1.0.0")
retriever = TfidfRetriever()
documents: dict[str, dict] = {}


class IngestRequest(BaseModel):
    text: str = Field(min_length=1)
    source: str = Field(default="api", max_length=200)


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    k: int = Field(default=4, ge=1, le=10)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "chunks": len(retriever.chunks),
        "documents": len(documents),
        "mode": "generative" if generative_available() else "extractive",
    }


@app.post("/ingest")
def ingest(req: IngestRequest) -> dict:
    chunks = chunk_text(req.text)
    if not chunks:
        raise HTTPException(status_code=422, detail="No indexable text found.")
    doc_id = f"doc-{uuid.uuid4().hex[:8]}"
    for chunk in chunks:
        retriever.add(chunk, source=req.source, doc_id=doc_id)
    documents[doc_id] = {"source": req.source, "chunks": len(chunks)}
    return {"doc_id": doc_id, "chunks": len(chunks), "source": req.source}


@app.post("/ingest-file")
async def ingest_file(file: UploadFile) -> dict:
    raw = await file.read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=422, detail="Only UTF-8 text/markdown files are supported.")
    return ingest(IngestRequest(text=text, source=file.filename or "upload"))


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    contexts = retriever.search(req.question, k=req.k)
    if generative_available() and contexts:
        answer = generate_answer(req.question, contexts)
        mode = "generative"
    else:
        answer = extractive_answer(req.question, contexts)
        mode = "extractive"
    return {
        "answer": answer,
        "mode": mode,
        "citations": [
            {"source": c["source"], "chunk_id": c["chunk_id"], "score": c["score"]}
            for c in contexts
        ],
    }


@app.get("/documents")
def list_documents() -> dict:
    return {"documents": documents}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(BASE_DIR / "static" / "index.html")


app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
