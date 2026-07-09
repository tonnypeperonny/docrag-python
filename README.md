# DocRag (Python version)

Same project as the .NET version in `D:\proj\docrag-dotnet` — hybrid semantic
search + RAG over local documents — rebuilt in Python for learning. Every file
mirrors a C# file and is heavily commented with the Python-vs-C# differences.

| Python | C# | What it does |
|---|---|---|
| `chunker.py` | `Chunker.cs` | Paragraph-aware splitting with overlap |
| `embedding_service.py` | `EmbeddingService.cs` | all-MiniLM-L6-v2 embeddings |
| `search_index.py` | `SearchIndex.cs` | ES index, bulk ingest, BM25 + kNN + RRF |
| `answer_service.py` | `AnswerService.cs` | Claude grounded answering |
| `ollama_service.py` | `OllamaAnswerService.cs` | Local-model answering via Ollama (keyless) |
| `main.py` | `Program.cs` | CLI entry point |

## Setup & usage

```powershell
python -m venv .venv                      # venv ≈ per-project packages (like a local NuGet cache)
.\.venv\Scripts\pip install -r requirements.txt

.\.venv\Scripts\python main.py ingest sample-docs
.\.venv\Scripts\python main.py search which days can I work from home
.\.venv\Scripts\python main.py ask what is the training budget
```

`ask` uses Claude when `ANTHROPIC_API_KEY` is set, otherwise a free local model
via [Ollama](https://ollama.com) (`ollama pull llama3.2:3b` once). Override with
`DOCRAG_LLM=claude|ollama`; pick the local model with `DOCRAG_LLM_MODEL`.

Needs the same Elasticsearch container as the .NET version (`docrag-es` on :9200).
Uses its own index (`doc-chunks-py`) so both versions coexist.

## The interesting differences (.NET vs Python)

1. **Embeddings: 80 lines vs 3.** C# hand-rolls the pipeline (WordPiece tokenizer,
   ONNX forward pass, mean pooling, L2 normalize) because .NET has no
   sentence-transformers equivalent. Python: `SentenceTransformer(...).encode(text)`.
   This is Python's real advantage in ML — the ecosystem, not the language.

2. **Typed client vs raw dicts.** The C# Elasticsearch client builds queries with
   typed descriptors — the compiler catches a wrong field name. The Python client
   takes plain dicts that map 1:1 to the ES JSON API — nothing is checked until
   the server rejects it. Flexibility vs safety; you'll meet this tradeoff all
   over Python.

3. **Model distribution.** .NET needed a download script for the ONNX file;
   sentence-transformers auto-downloads to `~/.cache/huggingface` on first use.

4. **Language mappings you'll keep using:**
   | C# | Python |
   |---|---|
   | `record` | `@dataclass(frozen=True)` |
   | LINQ (`Select`/`Where`) | list comprehensions / generator expressions |
   | `yield return` iterator | generator function (`yield`) |
   | `switch` expression | `match` statement |
   | `$"..."` interpolation | f-strings `f"..."` |
   | `string.Join(sep, xs)` | `sep.join(xs)` |
   | `buffer[^1]` | `buffer[-1]` |
   | `private` field | `_underscore` naming convention (not enforced) |
   | `Main()` | `if __name__ == "__main__":` |

5. **Same output.** Both versions produce identical rankings and RRF scores for
   the same query — same model, same algorithm, different language.
