"""Retrieval for questions that need more than one search.

Plan the question into sub-questions, retrieve for each, then merge the
rankings into a single context window.

The merge is round-robin — rank 1 of every sub-question, then rank 2 of every
sub-question, and so on — NOT another Reciprocal Rank Fusion pass like
`search_index.hybrid_search`. That is the whole point of doing it here. RRF
rewards chunks that several rankings agree on, which is right when the rankings
are two views of the *same* query; here the rankings answer *different*
questions, and the chunk that only one sub-question found is exactly the one
that must survive. Round-robin guarantees every part of the question is
represented in the window; RRF would let a strong first part crowd out a weak
second one, which is the failure being fixed.

With a single sub-question the merge collapses to the plain ranking, so
`--decompose` on a simple question is identical to not passing it at all.
"""

from dataclasses import dataclass
from math import ceil

from query_planner import OllamaQueryPlanner, QueryPlan
from search_index import ScoredChunk, SearchIndex


@dataclass(frozen=True)
class DecomposedRetrieval:
    chunks: list[ScoredChunk]                       # the merged context
    plan: QueryPlan
    per_sub: list[tuple[str, list[ScoredChunk]]]    # trace: what each part found

    @property
    def sub_questions(self) -> tuple[str, ...]:
        return self.plan.sub_questions


def _chunk_id(chunk: ScoredChunk) -> str:
    return f"{chunk.source_file}#{chunk.ordinal}"


def merge_round_robin(per_sub: list[tuple[str, list[ScoredChunk]]],
                      top_n: int) -> list[ScoredChunk]:
    """Interleave the per-sub-question rankings, best ranks first.

    Each sub-question gets `ceil(top_n / n)` slots. Sub-questions that return
    fewer results than that (or whose results were already taken by an earlier
    one) leave slots free, so a top-up pass fills the rest in rank order rather
    than shipping a short context.
    """
    budget = ceil(top_n / len(per_sub))

    merged: list[ScoredChunk] = []
    seen: set[str] = set()

    for depth in range(budget):
        for _, results in per_sub:
            if depth >= len(results):
                continue
            chunk = results[depth]
            if _chunk_id(chunk) not in seen:
                seen.add(_chunk_id(chunk))
                merged.append(chunk)

    for chunk in (c for _, results in per_sub for c in results):
        if len(merged) >= top_n:
            break
        if _chunk_id(chunk) not in seen:
            seen.add(_chunk_id(chunk))
            merged.append(chunk)

    return merged[:top_n]


def search_decomposed(index: SearchIndex, planner: OllamaQueryPlanner, question: str,
                      mode: str = "hybrid", top_n: int = 5) -> DecomposedRetrieval:
    plan = planner.plan(question)
    per_sub = [
        (sub, index.search(sub, mode=mode, top_n=top_n))
        for sub in plan.sub_questions
    ]
    return DecomposedRetrieval(
        chunks=merge_round_robin(per_sub, top_n),
        plan=plan,
        per_sub=per_sub,
    )
