"""Question decomposition with a local model via Ollama.

A question like "what are the latency targets for Atlas and Beacon" gets one
embedding and one BM25 query, and whichever half of it is lexically loudest can
take the whole top-k window. Splitting it first gives each part its own search.

This is the planner half: question in, sub-questions out. It deliberately does
NOT touch the index — `decomposed_search.py` owns the retrieve-and-merge step.

Runs on the same local Ollama as answering, but the model is separately
selectable (`DOCRAG_PLANNER_MODEL`) so a run can pair a small planner with a
bigger answerer, or the other way round, without a code change.
"""

import json
import os
from dataclasses import dataclass

import requests

from ollama_service import MODEL, OLLAMA_URL

PLANNER_MODEL = os.environ.get("DOCRAG_PLANNER_MODEL", MODEL)

# Each sub-question costs a full retrieval, and the merged context still has to
# fit the answer model's window — so cap the fan-out.
MAX_SUB_QUESTIONS = 4

# The examples are deliberately about things that appear in NO sample document
# and in neither eval set. Demonstrating the split on a real corpus question
# would be teaching to the test — the planner has to generalise the shape of
# the operation, not memorise our answers.
SYSTEM_PROMPT = """\
You split a question into the separate searches a document search engine needs
to answer it completely.

Reply with JSON only, in exactly this shape:
{"needs_split": true, "sub_questions": ["...", "..."]}

Decide "needs_split" FIRST:
- true if the question names two different subjects to look up (two projects,
  two policies, two items), or asks two things that would be written down in
  two different places.
- false if one lookup answers the whole question.
- Never set it to true just to reword the question.

When "needs_split" is false, put the original question unchanged as the single
entry and stop.

When it is true:
- Each entry must be a complete, standalone question that makes sense with no
  other context — carry the subject over into every part.
- Reuse the wording of the original question. You are only re-cutting it, so
  never add a topic, term or angle it did not mention.
- Every entry must ask for a different fact. Never restate one entry in other
  words: if two parts would be answered by the same sentence of a document,
  they are one question, not two.
- At most 4 entries.

Examples:

Question: what is the office wifi password
{"needs_split": false, "sub_questions": ["what is the office wifi password"]}

Question: how long does the induction course take
{"needs_split": false, "sub_questions": ["how long does the induction course take"]}

Question: what are the opening hours of the London and Berlin offices
{"needs_split": true, "sub_questions": ["what are the opening hours of the London office", "what are the opening hours of the Berlin office"]}

Question: is the canteen open on Saturday and does it take card
{"needs_split": true, "sub_questions": ["is the canteen open on Saturday", "does the canteen take card"]}
"""


@dataclass(frozen=True)
class QueryPlan:
    question: str
    sub_questions: tuple[str, ...]

    @property
    def is_split(self) -> bool:
        return len(self.sub_questions) > 1


class OllamaQueryPlanner:
    def plan(self, question: str) -> QueryPlan:
        """Split `question`, or fall back to it unchanged.

        Every failure path — Ollama down, non-JSON output, wrong shape, empty
        list — returns the single-question plan, which makes the rest of the
        pipeline behave exactly as it does without `--decompose`. A planner
        that misbehaves should cost accuracy, never availability.
        """
        try:
            response = requests.post(
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model": PLANNER_MODEL,
                    "stream": False,
                    "format": "json",  # constrain the decoder to valid JSON
                    # Greedy decoding, fixed seed: the same question should
                    # produce the same plan, or the eval numbers move on their
                    # own. "should" — Ollama is not bit-deterministic even at
                    # temperature 0, so the decompose runs are repeated in
                    # EXPERIMENTS.md rather than trusted from one sample.
                    "options": {"temperature": 0, "seed": 0},
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"Question: {question}"},
                    ],
                },
                timeout=300,  # first call loads the model into VRAM
            )
            response.raise_for_status()

            raw = json.loads(response.json()["message"]["content"])
            # The model has to commit to splitting before it starts writing
            # parts. Without that decision up front a small model happily
            # "splits" a single question into paraphrases of itself, which
            # costs a retrieval and buys nothing.
            if not raw.get("needs_split"):
                return QueryPlan(question, (question,))
            subs = _clean(raw["sub_questions"])
        except (requests.RequestException, ValueError, KeyError, TypeError):
            return QueryPlan(question, (question,))

        return QueryPlan(question, tuple(subs[:MAX_SUB_QUESTIONS]) or (question,))


def _clean(sub_questions: object) -> list[str]:
    """Keep non-empty strings, trimmed, de-duplicated case-insensitively."""
    if not isinstance(sub_questions, list):
        raise TypeError("sub_questions is not a list")

    cleaned: list[str] = []
    seen: set[str] = set()
    for item in sub_questions:
        if not isinstance(item, str) or not item.strip():
            continue
        text = item.strip()
        if text.lower() in seen:
            continue
        seen.add(text.lower())
        cleaned.append(text)
    return cleaned
