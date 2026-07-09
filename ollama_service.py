"""Grounded answer generation with a LOCAL model via Ollama.

Ollama (https://ollama.com) runs open-weight models (Llama, Qwen, Mistral...)
on your own GPU/CPU and exposes a small HTTP API on http://localhost:11434.
No API key, no cost per request, nothing leaves your machine.

Same contract as answer_service.py (Claude): takes the retrieved chunks,
returns a cited answer. Two RAG-relevant differences you will notice:

1. Quality: a 3B local model follows the "cite your sources, don't invent"
   instruction less reliably than a frontier model. Try a stronger local
   model (`ollama pull qwen2.5:7b`) and compare — great interview talking
   point about the quality/cost/privacy tradeoff in RAG systems.
2. Latency: first call is slow (model loads into VRAM), then it stays warm.
"""

import os

import requests  # the de-facto standard HTTP library (like HttpClient in .NET)

from search_index import ScoredChunk

# os.environ.get(key, default) — like Environment.GetEnvironmentVariable
# with a fallback. Lets you switch models without touching code:
#   $env:DOCRAG_LLM_MODEL = "qwen2.5:7b"
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

        # Ollama's /api/chat mirrors the shape most chat APIs use:
        # a list of {role, content} messages. `stream: False` means
        # "give me the whole answer in one JSON response" instead of
        # token-by-token server-sent events.
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
            timeout=300,  # first call loads the model into VRAM — be patient
        )
        response.raise_for_status()  # turn HTTP 4xx/5xx into an exception

        # .json() parses the response body; the answer lives at message.content
        return response.json()["message"]["content"]
