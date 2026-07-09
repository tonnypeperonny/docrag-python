"""docrag — hybrid semantic search + RAG over local documents (Python version).

Same architecture as the .NET version in `docrag/src/DocRag`:

    ingest:  files -> chunker -> MiniLM embeddings -> Elasticsearch
    search:  BM25 + kNN in parallel -> Reciprocal Rank Fusion -> top chunks
    ask:     top chunks + question -> Claude -> cited answer

Usage (from D:\\llm\\docrag, with the venv activated or via .venv\\Scripts\\python.exe):

    python main.py ingest sample-docs
    python main.py search which days can I work from home
    python main.py ask what is the training budget and does it roll over

Environment:
    ES_URL             Elasticsearch endpoint (default http://localhost:9200)
    ANTHROPIC_API_KEY  required for `ask`
"""

import os
import sys
from pathlib import Path

from elasticsearch import Elasticsearch

import chunker
from answer_service import AnswerService
from embedding_service import EmbeddingService
from search_index import INDEX_NAME, SearchIndex


def cmd_ingest(index: SearchIndex, embedder: EmbeddingService, folder: str) -> None:
    root = Path(folder).resolve()   # pathlib.Path ≈ .NET's Path + DirectoryInfo in one
    # rglob = recursive glob. Two passes (one per extension), concatenated.
    files = sorted(list(root.rglob("*.md")) + list(root.rglob("*.txt")))
    print(f"Indexing {len(files)} file(s) from {root} into '{INDEX_NAME}'...")

    index.recreate()

    pairs = []  # list of (Chunk, embedding-vector) tuples
    for file in files:
        relative = str(file.relative_to(root))
        # read_text() = File.ReadAllText(); chunker.split is a generator,
        # the for-loop pulls chunks out of it lazily.
        for chunk in chunker.split(relative, file.read_text(encoding="utf-8")):
            pairs.append((chunk, embedder.embed(chunk.content)))

    index.index(pairs)
    print(f"Indexed {len(pairs)} chunk(s).")


def cmd_search(index: SearchIndex, query: str) -> None:
    for i, result in enumerate(index.hybrid_search(query), start=1):
        print(f"--- #{i}  rrf={result.score:.4f}  {result.source_file} (chunk {result.ordinal})")
        # Conditional expression = C# ternary, but reads middle-out:
        # value_if_true if condition else value_if_false
        preview = result.content[:300] + "…" if len(result.content) > 300 else result.content
        print(preview)
        print()


def cmd_ask(index: SearchIndex, question: str) -> int:
    context = index.hybrid_search(question)
    if not context:  # empty list is "falsy" — idiomatic emptiness check
        print("No indexed content matched the question. Run `python main.py ingest <folder>` first.")
        return 1

    retrieved = ", ".join(f"{c.source_file}#{c.ordinal}" for c in context)
    print(f"Retrieved {len(context)} chunk(s): {retrieved}\n")

    # Provider selection: Claude when an API key is present, otherwise the
    # free local model via Ollama. Force one with DOCRAG_LLM=claude|ollama.
    provider = os.environ.get(
        "DOCRAG_LLM",
        "claude" if os.environ.get("ANTHROPIC_API_KEY") else "ollama",
    )
    if provider == "claude":
        print("[answering with Claude]\n")
        print(AnswerService().ask(question, context))
    else:
        # Import here (not at the top) so this path works even if the
        # anthropic package's needs aren't met — "lazy import" is a
        # common Python pattern for optional dependencies.
        from ollama_service import MODEL, OllamaAnswerService

        print(f"[answering with local model {MODEL} via Ollama]\n")
        print(OllamaAnswerService().ask(question, context))
    return 0


def cmd_chunks(index: SearchIndex, full: bool) -> None:
    """Inspect what ingestion actually stored — chunk text AND embedding.

    Useful for debugging retrieval: is the chunking sane? Are the vectors
    unit-length (they must be, for cosine similarity to behave)?
    """
    docs = index.list_all()
    print(f"{len(docs)} chunk(s) in '{INDEX_NAME}':\n")
    for doc in docs:
        embedding = doc["embedding"]
        # Show just the first 6 of 384 dims — the numbers themselves are
        # meaningless to humans; what matters is the shape and the norm.
        preview = ", ".join(f"{v:.4f}" for v in embedding[:6])
        # The vector norm should be ~1.0 because we normalize at embed time.
        # sum(v*v for v in ...) ** 0.5 — a generator expression feeding sum().
        norm = sum(v * v for v in embedding) ** 0.5
        content = doc["content"]
        print(f"=== {doc['source_file']} #{doc['ordinal']}  ({len(content)} chars)")
        print(f"    embedding[{len(embedding)}] = [{preview}, …]  |v|={norm:.3f}")
        print(content if full else (content[:200] + "…" if len(content) > 200 else content))
        print()


def main() -> int:
    # sys.argv is the raw argument list; argv[0] is the script name itself.
    # (For anything bigger, use the stdlib `argparse` module instead.)
    # `chunks` needs no argument, every other command does.
    if len(sys.argv) < 2 or (len(sys.argv) < 3 and sys.argv[1].lower() != "chunks"):
        print(__doc__)  # the module docstring at the top doubles as help text
        return 1

    command = sys.argv[1].lower()
    argument = " ".join(sys.argv[2:])   # the rest of the args, joined back up

    client = Elasticsearch(os.environ.get("ES_URL", "http://localhost:9200"))
    embedder = EmbeddingService()
    index = SearchIndex(client, embedder)

    # Python 3.10+ structural pattern matching — close cousin of C#'s
    # switch expression.
    match command:
        case "ingest":
            cmd_ingest(index, embedder, argument)
            return 0
        case "search":
            cmd_search(index, argument)
            return 0
        case "ask":
            return cmd_ask(index, argument)
        case "chunks":
            cmd_chunks(index, full="--full" in argument)
            return 0
        case _:
            print(f"Unknown command: {command}")
            return 1


# This guard means "only run main() when executed as a script, not when
# imported as a module" — Python's equivalent of a Main() entry point.
if __name__ == "__main__":
    raise SystemExit(main())
