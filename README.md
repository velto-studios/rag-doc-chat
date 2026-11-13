# rag-doc-chat

Chat with your own documents. A small, production-shaped RAG (retrieval-augmented
generation) service built with Python and FastAPI.

Built by [Velto Studios](https://veltostudios.github.io) as a working sample of
how we build AI systems: retrieval first, citations always, and no magic that
only works in a demo.

## What it does

- Ingests UTF-8 text / Markdown documents through a REST API or file upload
- Splits them into overlapping, paragraph-aware chunks
- Retrieves the most relevant chunks for a question with a pure-Python TF-IDF
  index (no model downloads, no external vector DB)
- Answers with citations in one of two modes:
  - **Extractive mode** (default, fully offline): quotes the most relevant
    passages with source citations
  - **Generative mode**: set `LLM_API_KEY` and answers are synthesized by any
    OpenAI-compatible chat API, grounded in the retrieved passages
- Ships a minimal web UI at `/` plus a typed JSON API

## Architecture

```
            ┌──────────────┐   chunks    ┌────────────────────┐
 POST /ingest ──► chunking.py ──────────► │  TfidfRetriever     │
            │              │             │  (in-memory index)  │
            └──────────────┘             └─────────▲──────────┘
                                                   │ top-k + scores
 POST /ask ──► retriever.search(query) ────────────┘
                     │
                     ▼
            ┌────────────────────┐   LLM_API_KEY set?   ┌──────────────────┐
            │  extractive answer │ ◄──── no ───────────►│  OpenAI-compat    │
            │  (quote passages)  │ ──── yes ──────────► │  chat completion  │
            └────────────────────┘                       └──────────────────┘
                     │
                     ▼
        answer + citations [{source, chunk_id, score}]
```

Design choices, deliberately:

- **Pure-Python TF-IDF** keeps the demo dependency-light and runnable anywhere.
  The retriever interface (`add` / `search`) is the seam where you would swap
  in embeddings + a vector store (pgvector, Qdrant, ...) for a larger corpus.
- **Extractive fallback** means the service is honest out of the box: with no
  API key it still answers, grounded in retrieved text, and says so.
- **Citations on every answer** - each response lists the source document,
  chunk id and similarity score behind it.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
# open http://localhost:8000
```

Or with Docker:

```bash
docker build -t rag-doc-chat . && docker run -p 8000:8000 rag-doc-chat
```

Generative mode (any OpenAI-compatible endpoint):

```bash
export LLM_API_KEY=sk-...            # required
export LLM_BASE_URL=https://api.openai.com/v1   # optional, this is the default
export LLM_MODEL=gpt-4o-mini                    # optional
```

## API

| Endpoint | Description |
|---|---|
| `GET /health` | service status, index size, answering mode |
| `POST /ingest` | `{text, source}` - index a document |
| `POST /ingest-file` | multipart upload of a `.txt` / `.md` file |
| `POST /ask` | `{question, k}` - cited answer |
| `GET /documents` | indexed documents |

## Demo (actual output, extractive mode)

After ingesting the two files in `samples/`:

```
$ curl -s localhost:8000/ask -H 'Content-Type: application/json' \
    -d '{"question":"what stack does the studio use for AI work?","k":2}'

"answer": "Based on the indexed documents:

[1] # Velto Studios - Services Overview ... The stack is Python and FastAPI
    on the backend, with OpenAI-compatible model APIs for generation. ...

(Extractive mode: no LLM_API_KEY set, so answers quote the most relevant
passages directly. Set LLM_API_KEY for synthesized answers.)",

"citations": [
  {"source": "velto-services.md",  "chunk_id": 0, "score": 0.8549},
  {"source": "onboarding-faq.md",  "chunk_id": 2, "score": 0.4491}
]
```

## Tests

```bash
pytest tests/ -q
# 8 passed
```

Covers chunking, retrieval ranking, and the full ingest - ask round trip
through the HTTP API.

## Layout

```
app/
  main.py        FastAPI routes and app wiring
  chunking.py    paragraph-aware text splitting with overlap
  retriever.py   pure-Python TF-IDF index with cosine search
  llm.py         extractive answers + optional OpenAI-compatible generation
static/index.html  minimal chat UI
samples/           two Markdown files to try it with
tests/             pytest suite (offline)
```

## License

MIT
