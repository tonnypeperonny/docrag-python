# DocRag (Python version)

Hybrid semantic search + RAG over local documents — Python port of the .NET
version in `D:\proj\docrag-dotnet`. Both share the same pipeline:

    ingest:  files -> chunker -> MiniLM embeddings -> Elasticsearch
    search:  BM25 + kNN in parallel -> Reciprocal Rank Fusion -> top chunks
    ask:     top chunks + question -> local LLM via Ollama -> cited answer

Fully local: embeddings, search, and answering all run on this machine —
no API keys, nothing leaves the box.

| Module | What it does |
|---|---|
| `chunker.py` | Paragraph-aware splitting with overlap |
| `embedding_service.py` | all-MiniLM-L6-v2 embeddings (384 dims) |
| `search_index.py` | ES index, bulk ingest, BM25 + kNN + RRF |
| `ollama_service.py` | Grounded answering via a local model on Ollama |
| `eval_retrieval.py` + `evalset.jsonl` | Retrieval quality eval (hit@k, MRR) |
| `main.py` | CLI entry point |

## Setup & usage

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt

.\.venv\Scripts\python main.py ingest sample-docs
.\.venv\Scripts\python main.py search which days can I work from home
.\.venv\Scripts\python main.py ask what is the training budget
```

`ask` answers with a local model via [Ollama](https://ollama.com)
(`ollama pull llama3.2:3b` once). Pick the model with `DOCRAG_LLM_MODEL`.

Needs the same Elasticsearch container as the .NET version (`docrag-es` on :9200).
Uses its own index (`doc-chunks-py`) so both versions coexist — same model and
same algorithm, so both produce identical rankings for the same query.

## Experimenting

Retrieval is tunable per run: `--mode hybrid|bm25|knn` and `--top N` on
`search`/`ask`, `--chunk N` and `--no-overlap` on `ingest`. `eval` scores all
modes against the labelled questions in `evalset.jsonl` (hit@k and MRR, split
into keyword vs paraphrase questions):

```powershell
.\.venv\Scripts\python main.py eval
.\.venv\Scripts\python main.py search --mode knn when will I get my money back
```

Scenarios and results are tracked in [EXPERIMENTS.md](EXPERIMENTS.md).
