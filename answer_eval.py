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
number blindly. `kind` (single vs multi) shows whether the misses are on
single-fact questions or the multi-source ones that need two chunks combined.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from ollama_service import MODEL, OllamaAnswerService
from search_index import SearchIndex

ANSWERSET_PATH = Path(__file__).parent / "answerset.jsonl"

# Phrases that count as the model correctly declining to answer.
REFUSAL_MARKERS = (
    "do not contain",
    "does not contain",
    "not contain",
    "not in the sources",
    "sources do not",
    "does not mention",
    "do not mention",
    "no information",
    "not covered",
    "not provided",
    "cannot find",
    "could not find",
    "unable to find",
    "not available in",
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
                     top_n: int = 5, mode: str = "hybrid") -> list[AnswerOutcome]:
    outcomes = []
    for case in load_answerset():
        context = index.search(case.question, mode=mode, top_n=top_n)
        answer = service.ask(case.question, context)
        correct, missing = _score(case, answer)
        outcomes.append(AnswerOutcome(case, correct, missing, answer))
    return outcomes


def print_answer_report(outcomes: list[AnswerOutcome], top_n: int, mode: str) -> None:
    def rate(subset: list[AnswerOutcome]) -> str:
        if not subset:
            return "  -  "
        return f"{sum(o.correct for o in subset) / len(subset):.2f}"

    kinds = sorted({o.case.kind for o in outcomes})
    print(f"answer accuracy — model {MODEL}, mode {mode}, top {top_n}\n")
    print(f"{'overall':<10} {rate(outcomes)}   ({sum(o.correct for o in outcomes)}/{len(outcomes)})")
    for kind in kinds:
        subset = [o for o in outcomes if o.case.kind == kind]
        print(f"{kind:<10} {rate(subset)}   ({sum(o.correct for o in subset)}/{len(subset)})")

    failures = [o for o in outcomes if not o.correct]
    if failures:
        print("\nfailures:")
        for o in failures:
            snippet = " ".join(o.answer.split())
            snippet = snippet[:120] + "…" if len(snippet) > 120 else snippet
            print(f"  [{o.case.kind}] {o.case.question!r}")
            print(f"      missing: {', '.join(o.missing)}")
            print(f"      answer:  {snippet}")
