"""Split documents into overlapping chunks for retrieval."""
from __future__ import annotations


def chunk_text(text: str, max_chars: int = 550, overlap: int = 100) -> list[str]:
    """Split text into chunks of at most max_chars with overlap.

    Splits on paragraph boundaries first, then sentence-ish boundaries,
    so chunks stay semantically coherent.
    """
    text = text.strip()
    if not text:
        return []
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if len(para) > max_chars:
            # hard-split very long paragraphs on sentence boundaries
            sentences = para.replace(". ", ".\n").split("\n")
        else:
            sentences = [para]
        for piece in sentences:
            piece = piece.strip()
            if not piece:
                continue
            if current and len(current) + len(piece) + 1 > max_chars:
                chunks.append(current)
                current = current[-overlap:] if overlap else ""
                current = (current + " " + piece).strip()
            else:
                current = (current + "\n\n" + piece).strip() if current else piece
    if current:
        chunks.append(current)
    return chunks
