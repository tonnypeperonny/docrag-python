"""docrag — hybrid semantic search + RAG over local documents.

    ingest:  files -> chunker -> MiniLM embeddings -> Elasticsearch
    search:  BM25 + kNN in parallel -> Reciprocal Rank Fusion -> top chunks
    ask:     top chunks + question -> local LLM via Ollama -> cited answer

Usage:

    python main.py ingest sample-docs [--chunk N] [--no-overlap]
    python main.py search [--mode hybrid|bm25|knn] [--top N] which days can I work from home
    python main.py ask [--mode ...] [--top N] what is the training budget and does it roll over
    python main.py eval [--mode ...] [--k N]     score retrieval against evalset.jsonl
    python main.py answer-eval [--top N]         score generated answers against answerset.jsonl
    python main.py chunks [--full]               inspect stored chunks + embeddings

Experiment knobs:
    --mode      retrieval strategy (default hybrid); eval runs all modes unless given
    --top/--k   how many chunks to retrieve/judge (default 5 / 3)
    --chunk     target chunk size in chars at ingest time (default 1200)
    --no-overlap  disable paragraph overlap between chunks at ingest time
    --decompose   split the question into sub-questions and retrieve for each
                  (search, ask, answer-eval)
    --enumerate   answer with the prompt that requires every applicable rule
                  (ask, answer-eval)

Environment:
    ES_URL               Elasticsearch endpoint (default http://localhost:9200)
    OLLAMA_URL           Ollama endpoint (default http://localhost:11434)
    DOCRAG_LLM_MODEL     model name for `ask` (default llama3.2:3b)
    DOCRAG_PLANNER_MODEL model name for --decompose (defaults to DOCRAG_LLM_MODEL)
"""

import os
import sys
from pathlib import Path

from elasticsearch import Elasticsearch

import chunker
from answer_eval import evaluate_answers, print_answer_report
from decomposed_search import DecomposedRetrieval, search_decomposed
from embedding_service import EmbeddingService
from eval_retrieval import evaluate, print_report
from ollama_service import MODEL, SYSTEM_PROMPT_ENUMERATE, OllamaAnswerService
from query_planner import PLANNER_MODEL, OllamaQueryPlanner
from search_index import INDEX_NAME, MODES, SearchIndex


def pop_option(args: list[str], name: str) -> str | None:
    """Remove `name value` from args (in place); return the value or None."""
    if name not in args:
        return None
    i = args.index(name)
    if i + 1 >= len(args):
        raise SystemExit(f"{name} needs a value")
    args.pop(i)
    return args.pop(i)


def pop_switch(args: list[str], name: str) -> bool:
    """Remove the bare flag `name` from args (in place); return whether it was there."""
    if name not in args:
        return False
    args.remove(name)
    return True


def cmd_ingest(index: SearchIndex, embedder: EmbeddingService, folder: str,
               chunk_chars: int, overlap: bool) -> None:
    root = Path(folder).resolve()
    files = sorted(list(root.rglob("*.md")) + list(root.rglob("*.txt")))
    print(f"Indexing {len(files)} file(s) from {root} into '{INDEX_NAME}' "
          f"(chunk={chunk_chars}, overlap={overlap})...")

    index.recreate()

    pairs = []  # (Chunk, embedding-vector) tuples
    for file in files:
        relative = str(file.relative_to(root))
        for chunk in chunker.split(relative, file.read_text(encoding="utf-8"),
                                   target_chars=chunk_chars, overlap=overlap):
            pairs.append((chunk, embedder.embed(chunk.content)))

    index.index(pairs)
    print(f"Indexed {len(pairs)} chunk(s).")


def retrieve(index: SearchIndex, question: str, mode: str, top_n: int, decompose: bool):
    """One-shot retrieval, or plan -> retrieve per sub-question -> merge.

    Returns (chunks, DecomposedRetrieval | None) — the second value is the
    planner trace, and is None whenever --decompose was not asked for.
    """
    if not decompose:
        return index.search(question, mode=mode, top_n=top_n), None

    result = search_decomposed(index, OllamaQueryPlanner(), question, mode=mode, top_n=top_n)
    return result.chunks, result


def print_plan(result: DecomposedRetrieval) -> None:
    """Show what the planner did and what each part of the question retrieved."""
    count = len(result.sub_questions)
    print(f"[planner {PLANNER_MODEL} -> {count} sub-question(s)]")
    for sub, results in result.per_sub:
        found = ", ".join(f"{c.source_file}#{c.ordinal}" for c in results) or "nothing"
        print(f"  - {sub}")
        print(f"      {found}")
    print()


def cmd_search(index: SearchIndex, query: str, mode: str, top_n: int, decompose: bool) -> None:
    context, plan = retrieve(index, query, mode, top_n, decompose)
    if plan is not None:
        print_plan(plan)

    for i, result in enumerate(context, start=1):
        print(f"--- #{i}  {mode}={result.score:.4f}  {result.source_file} (chunk {result.ordinal})")
        preview = result.content[:300] + "…" if len(result.content) > 300 else result.content
        print(preview)
        print()


def cmd_ask(index: SearchIndex, question: str, mode: str, top_n: int,
            decompose: bool, enumerate_all: bool) -> int:
    context, plan = retrieve(index, question, mode, top_n, decompose)
    if plan is not None:
        print_plan(plan)

    if not context:
        print("No indexed content matched the question. Run `python main.py ingest <folder>` first.")
        return 1

    retrieved = ", ".join(f"{c.source_file}#{c.ordinal}" for c in context)
    print(f"Retrieved {len(context)} chunk(s): {retrieved}\n")

    service = OllamaAnswerService(SYSTEM_PROMPT_ENUMERATE) if enumerate_all else OllamaAnswerService()
    print(f"[answering with local model {MODEL} via Ollama]\n")
    print(service.ask(question, context))
    return 0


def cmd_eval(index: SearchIndex, mode: str | None, k: int) -> None:
    """Score retrieval against the labelled question set, per mode."""
    modes = [mode] if mode else list(MODES)
    print_report([evaluate(index, m, k=k) for m in modes])


def cmd_answer_eval(index: SearchIndex, mode: str, top_n: int,
                    decompose: bool, enumerate_all: bool) -> None:
    """Score end-to-end answers (retrieve + generate) against answerset.jsonl.

    The two workflow flags are independent on purpose: --decompose changes what
    reaches the context window, --enumerate changes what the model does with
    it. Running them separately is what tells you which one earned the win.
    """
    service = OllamaAnswerService(SYSTEM_PROMPT_ENUMERATE) if enumerate_all else OllamaAnswerService()
    workflow = "+".join(
        [name for name, on in (("decompose", decompose), ("enumerate", enumerate_all)) if on]
    ) or "one-shot"

    outcomes = evaluate_answers(index, service, top_n=top_n, mode=mode, decompose=decompose)
    print_answer_report(outcomes, top_n=top_n, mode=mode, workflow=workflow)


def cmd_chunks(index: SearchIndex, full: bool) -> None:
    """Inspect what ingestion actually stored — chunk text AND embedding.

    Useful for debugging retrieval: is the chunking sane? Are the vectors
    unit-length (they must be, for cosine similarity to behave)?
    """
    docs = index.list_all()
    print(f"{len(docs)} chunk(s) in '{INDEX_NAME}':\n")
    for doc in docs:
        embedding = doc["embedding"]
        # First 6 of 384 dims — what matters is the shape and the norm
        # (~1.0, because we normalize at embed time).
        preview = ", ".join(f"{v:.4f}" for v in embedding[:6])
        norm = sum(v * v for v in embedding) ** 0.5
        content = doc["content"]
        print(f"=== {doc['source_file']} #{doc['ordinal']}  ({len(content)} chars)")
        print(f"    embedding[{len(embedding)}] = [{preview}, …]  |v|={norm:.3f}")
        print(content if full else (content[:200] + "…" if len(content) > 200 else content))
        print()


def main() -> int:
    args = sys.argv[1:]

    # Pull the experiment flags out first; whatever is left after the command
    # name is the query/folder text.
    mode = pop_option(args, "--mode")
    top = pop_option(args, "--top")
    k = pop_option(args, "--k")
    chunk_chars = pop_option(args, "--chunk")
    no_overlap = pop_switch(args, "--no-overlap")
    full = pop_switch(args, "--full")
    decompose = pop_switch(args, "--decompose")
    enumerate_all = pop_switch(args, "--enumerate")

    if mode is not None and mode not in MODES:
        print(f"Unknown --mode '{mode}', expected one of {MODES}")
        return 1

    # `chunks`, `eval` and `answer-eval` need no argument, every other command does.
    command = args[0].lower() if args else ""
    if not command or (len(args) < 2 and command not in ("chunks", "eval", "answer-eval")):
        print(__doc__)  # the module docstring doubles as help text
        return 1

    argument = " ".join(args[1:])

    client = Elasticsearch(os.environ.get("ES_URL", "http://localhost:9200"))
    embedder = EmbeddingService()
    index = SearchIndex(client, embedder)

    match command:
        case "ingest":
            cmd_ingest(index, embedder, argument,
                       chunk_chars=int(chunk_chars or chunker.TARGET_CHARS),
                       overlap=not no_overlap)
            return 0
        case "search":
            cmd_search(index, argument, mode=mode or "hybrid", top_n=int(top or 5),
                       decompose=decompose)
            return 0
        case "ask":
            return cmd_ask(index, argument, mode=mode or "hybrid", top_n=int(top or 5),
                           decompose=decompose, enumerate_all=enumerate_all)
        case "eval":
            cmd_eval(index, mode, k=int(k or 3))
            return 0
        case "answer-eval":
            cmd_answer_eval(index, mode=mode or "hybrid", top_n=int(top or 5),
                            decompose=decompose, enumerate_all=enumerate_all)
            return 0
        case "chunks":
            cmd_chunks(index, full=full)
            return 0
        case _:
            print(f"Unknown command: {command}")
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
