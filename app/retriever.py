"""Pure-Python TF-IDF retriever. No external services, no model downloads."""
from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class TfidfRetriever:
    """In-memory TF-IDF index over document chunks."""

    def __init__(self) -> None:
        self.chunks: list[dict] = []          # {doc_id, chunk_id, text, source}
        self._tf: list[Counter] = []
        self._df: Counter = Counter()
        self._n_docs = 0

    def add(self, text: str, source: str, doc_id: str | None = None) -> int:
        """Add a chunk; returns its chunk id."""
        tokens = tokenize(text)
        if not tokens:
            return -1
        chunk_id = len(self.chunks)
        self.chunks.append({
            "doc_id": doc_id or source,
            "chunk_id": chunk_id,
            "text": text,
            "source": source,
        })
        tf = Counter(tokens)
        self._tf.append(tf)
        for tok in tf:
            self._df[tok] += 1
        self._n_docs = len(self.chunks)
        return chunk_id

    def _idf(self, tok: str) -> float:
        return math.log((1 + self._n_docs) / (1 + self._df.get(tok, 0))) + 1.0

    def _vector(self, tf: Counter) -> dict[str, float]:
        norm = math.sqrt(sum(v * v for v in tf.values())) or 1.0
        return {t: (c / norm) * self._idf(t) for t, c in tf.items()}

    def search(self, query: str, k: int = 4) -> list[dict]:
        """Return the top-k chunks with cosine similarity scores."""
        if not self.chunks:
            return []
        qv = self._vector(Counter(tokenize(query)))
        scored = []
        for chunk, tf in zip(self.chunks, self._tf):
            dv = self._vector(tf)
            dot = sum(qv.get(t, 0.0) * w for t, w in dv.items())
            if dot > 0:
                scored.append((dot, chunk))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {"score": round(s, 4), "text": c["text"], "source": c["source"],
             "chunk_id": c["chunk_id"]}
            for s, c in scored[:k]
        ]
