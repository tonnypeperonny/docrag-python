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


class OllamaAnswerService:
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
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
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
