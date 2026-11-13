"""Optional generative answering via any OpenAI-compatible chat API.

If LLM_API_KEY is not set, the app runs in extractive mode: answers are
built from the retrieved passages themselves, with citations. That mode
works fully offline and is what the tests exercise.
"""
from __future__ import annotations

import os

import httpx

LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")


def generative_available() -> bool:
    return bool(LLM_API_KEY)


def generate_answer(question: str, contexts: list[dict]) -> str:
    """Call an OpenAI-compatible chat completion grounded in contexts."""
    context_block = "\n\n".join(
        f"[{i+1}] (source: {c['source']})\n{c['text']}"
        for i, c in enumerate(contexts)
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You answer questions using only the provided context passages. "
                "Cite passages like [1], [2] after each claim. If the context "
                "does not contain the answer, say so plainly."
            ),
        },
        {
            "role": "user",
            "content": f"Context:\n{context_block}\n\nQuestion: {question}",
        },
    ]
    resp = httpx.post(
        f"{LLM_BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {LLM_API_KEY}"},
        json={"model": LLM_MODEL, "messages": messages, "temperature": 0.2},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def extractive_answer(question: str, contexts: list[dict]) -> str:
    """Build a grounded answer from the retrieved passages alone."""
    if not contexts:
        return (
            "I could not find anything relevant in the indexed documents. "
            "Try ingesting more documents or rephrasing the question."
        )
    lines = ["Based on the indexed documents:\n"]
    for i, c in enumerate(contexts):
        snippet = " ".join(c["text"].split())
        if len(snippet) > 400:
            snippet = snippet[:400].rsplit(" ", 1)[0] + "..."
        lines.append(f"[{i+1}] {snippet}")
    lines.append(
        "\n(Extractive mode: no LLM_API_KEY set, so answers quote the most "
        "relevant passages directly. Set LLM_API_KEY for synthesized answers.)"
    )
    return "\n\n".join(lines)
