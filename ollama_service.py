"""Grounded answer generation with a local model via Ollama.

The retrieved chunks are passed as numbered sources; the system prompt
confines the model to those sources and requires citations — that
combination is what makes this RAG rather than "ask an LLM and hope".
Served by an open-weight model on http://localhost:11434: no API key,
no cost per request, nothing leaves the machine.
"""

import os

import requests

from search_index import ScoredChunk

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL = os.environ.get("DOCRAG_LLM_MODEL", "llama3.2:3b")

SYSTEM_PROMPT = """\
You are a documentation assistant. Answer the user's question using ONLY the
numbered sources provided in the message. Cite sources inline as [1], [2], etc.
If the sources do not contain the answer, say so explicitly — do not invent facts.
"""

# Same grounding rules, plus an explicit completeness requirement. The default
# prompt lets the model stop at the first matching rule, which is fine until a
# question has two valid answers in two different documents (meal limits, say)
# — then it reports the top-ranked one and silently drops the other.
SYSTEM_PROMPT_ENUMERATE = """\
You are a documentation assistant. Answer the user's question using ONLY the
numbered sources provided in the message. Cite sources inline as [1], [2], etc.
If the sources do not contain the answer, say so explicitly — do not invent facts.

If more than one rule, limit or figure in the sources applies to the question,
state EVERY one of them and say which situation each one covers. Never report
only the first match, and never merge two different figures into a range.
"""


class OllamaAnswerService:
    def __init__(self, system_prompt: str = SYSTEM_PROMPT) -> None:
        self._system_prompt = system_prompt

    def ask(self, question: str, context: list[ScoredChunk]) -> str:
        sources = "\n\n".join(
            f"[{i}] (from {chunk.source_file})\n{chunk.content}"
            for i, chunk in enumerate(context, start=1)
        )

        response = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": MODEL,
                "stream": False,
                # Greedy decoding, fixed seed. Sampling made `answer-eval` move
                # by a question or two between identical runs, which is larger
                # than most of the effects being measured — an A/B you cannot
                # reproduce is not a measurement.
                "options": {"temperature": 0, "seed": 0},
                "messages": [
                    {"role": "system", "content": self._system_prompt},
                    {
                        "role": "user",
                        "content": f"Sources:\n\n{sources}\n\nQuestion: {question}",
                    },
                ],
            },
            timeout=300,  # first call loads the model into VRAM
        )
        response.raise_for_status()

        return response.json()["message"]["content"]
