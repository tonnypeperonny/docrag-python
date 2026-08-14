"""End-to-end answer evaluation.

`eval_retrieval.py` only asks "did the right document come back". This asks
the harder question: "did the generated answer actually state the correct
fact". It runs the full pipeline per question (retrieve -> Ollama) and checks
the answer text against hand-labelled gold facts in `answerset.jsonl`.

Scoring is deliberately simple and transparent — normalized substring match,
no judge model:

- `facts`: a list of requirements; each requirement is a list of acceptable
  phrasings. The answer is correct only if EVERY requirement is met by at
  least one of its phrasings. Numbers are matched comma-insensitively so
  "4,000" and "4000" are the same.
- `must_refuse`: for questions whose answer is in no document — correct means
  the model declined ("not in the sources") instead of inventing one.

Substring matching is crude on purpose: it can false-negative when the model
phrases a fact in words we didn't list, so treat the score as fact-recall
with a known floor, and read the printed failures rather than trusting the
number blindly. Refusals are the worst case — "couldn't find" and "none of the
sources contain" both had to be added to REFUSAL_MARKERS after being scored as
hallucinations, so a must_refuse failure is worth reading before believing.
`kind` (single vs multi) shows whether the misses are on single-fact questions
or the multi-source ones that need two chunks combined.

The same harness scores the workflow variants: `decompose` swaps one-shot
retrieval for plan-and-merge (`decomposed_search.py`), and the caller chooses
the system prompt by handing in a pre-built service. Both default to off, so
the numbers stay comparable with earlier runs.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from decomposed_search import search_decomposed
from ollama_service import MODEL, OllamaAnswerService
from query_planner import OllamaQueryPlanner
from search_index import SearchIndex

ANSWERSET_PATH = Path(__file__).parent / "answerset.jsonl"

# Phrases that count as the model correctly declining to answer. Contractions
# need their own entries — "couldn't find" was scored as a hallucination for a
# while purely because only "could not find" was listed. Only consulted for
# `must_refuse` cases, so a generous list costs nothing elsewhere.
REFUSAL_MARKERS = (
    "do not contain",
    "does not contain",
    "not contain",
    # The negation does not always sit next to the verb: "none of the provided
    # sources contain ..." is a refusal that no "not <verb>" marker catches.
    "none of the",
    "no source",
    "not in the sources",
    "sources do not",
    "does not mention",
    "do not mention",
    "no mention",
    "no information",
    "no details",
    "not covered",
    "not provided",
    "cannot find",
    "could not find",
    "couldn't find",
    "did not find",
    "didn't find",
    "unable to find",
    "not available in",
    "does not specify",
    "do not specify",
    "doesn't specify",
    "don't specify",
)


@dataclass(frozen=True)
class AnswerCase:
    question: str
    kind: str                          # "single" | "multi"
    facts: tuple[tuple[str, ...], ...]  # requirements; each is a list of alternatives
    must_refuse: bool


@dataclass(frozen=True)
class AnswerOutcome:
    case: AnswerCase
    correct: bool
    missing: list[str]  # human-readable requirements the answer failed
    answer: str
    sub_questions: tuple[str, ...] = ()  # empty unless the run used --decompose


def _normalize(text: str) -> str:
    """Lowercase, drop thousands-separator commas, collapse whitespace."""
    return " ".join(text.lower().replace(",", "").split())


def load_answerset(path: Path = ANSWERSET_PATH) -> list[AnswerCase]:
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        facts = tuple(tuple(req) for req in raw.get("facts", []))
        cases.append(AnswerCase(
            question=raw["question"],
            kind=raw.get("kind", "single"),
            facts=facts,
            must_refuse=raw.get("must_refuse", False),
        ))
    return cases


def _score(case: AnswerCase, answer: str) -> tuple[bool, list[str]]:
    norm = _normalize(answer)
    if case.must_refuse:
        refused = any(marker in norm for marker in REFUSAL_MARKERS)
        return refused, ([] if refused else ["(expected the model to decline)"])

    missing = [
        " / ".join(req)
        for req in case.facts
        if not any(_normalize(alt) in norm for alt in req)
    ]
    return not missing, missing


def evaluate_answers(index: SearchIndex, service: OllamaAnswerService,
                     top_n: int = 5, mode: str = "hybrid",
                     decompose: bool = False) -> list[AnswerOutcome]:
    """Run the pipeline per question and score the answer.

    `service` arrives pre-built so the caller decides which system prompt is in
    play; `decompose` switches retrieval from one-shot to plan-and-merge. With
    both left at their defaults this is byte-for-byte the original path, so old
    numbers stay reproducible.
    """
    planner = OllamaQueryPlanner() if decompose else None

    outcomes = []
    for case in load_answerset():
        if planner is None:
            context, sub_questions = index.search(case.question, mode=mode, top_n=top_n), ()
        else:
            retrieval = search_decomposed(index, planner, case.question, mode=mode, top_n=top_n)
            context, sub_questions = retrieval.chunks, retrieval.sub_questions

        answer = service.ask(case.question, context)
        correct, missing = _score(case, answer)
        outcomes.append(AnswerOutcome(case, correct, missing, answer, sub_questions))
    return outcomes


def print_answer_report(outcomes: list[AnswerOutcome], top_n: int, mode: str,
                        workflow: str = "one-shot") -> None:
    def rate(subset: list[AnswerOutcome]) -> str:
        if not subset:
            return "  -  "
        return f"{sum(o.correct for o in subset) / len(subset):.2f}"

    kinds = sorted({o.case.kind for o in outcomes})
    print(f"answer accuracy — model {MODEL}, mode {mode}, top {top_n}, workflow {workflow}\n")
    print(f"{'overall':<10} {rate(outcomes)}   ({sum(o.correct for o in outcomes)}/{len(outcomes)})")
    for kind in kinds:
        subset = [o for o in outcomes if o.case.kind == kind]
        print(f"{kind:<10} {rate(subset)}   ({sum(o.correct for o in subset)}/{len(subset)})")

    # Cost side of the ledger: a planner that shreds every question into four
    # triples the retrievals, so accuracy alone would flatter it.
    if any(o.sub_questions for o in outcomes):
        counts = [len(o.sub_questions) for o in outcomes]
        split = [o for o in outcomes if len(o.sub_questions) > 1]
        print(f"\nsub-questions  mean {sum(counts) / len(counts):.2f}   "
              f"({len(split)}/{len(outcomes)} questions split)")
        for o in split:
            print(f"  [{o.case.kind}] {o.case.question!r}")
            for sub in o.sub_questions:
                print(f"      -> {sub}")

    failures = [o for o in outcomes if not o.correct]
    if failures:
        print("\nfailures:")
        for o in failures:
            snippet = " ".join(o.answer.split())
            snippet = snippet[:120] + "…" if len(snippet) > 120 else snippet
            print(f"  [{o.case.kind}] {o.case.question!r}")
            print(f"      missing: {', '.join(o.missing)}")
            print(f"      answer:  {snippet}")
