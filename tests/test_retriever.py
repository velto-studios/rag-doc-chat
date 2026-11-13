from app.chunking import chunk_text
from app.retriever import TfidfRetriever


def test_chunk_text_splits_and_keeps_content():
    text = "First paragraph about retrieval.\n\nSecond paragraph about FastAPI."
    chunks = chunk_text(text)
    assert chunks
    joined = " ".join(chunks)
    assert "retrieval" in joined and "FastAPI" in joined


def test_chunk_text_empty():
    assert chunk_text("") == []
    assert chunk_text("   \n\n ") == []


def test_retriever_ranks_relevant_chunk_first():
    r = TfidfRetriever()
    r.add("RAG grounds answers in retrieved documents.", source="a.md")
    r.add("The weather in Kanpur is hot in May.", source="b.md")
    hits = r.search("what does RAG do with documents", k=1)
    assert hits and hits[0]["source"] == "a.md"


def test_retriever_empty_index():
    assert TfidfRetriever().search("anything") == []
