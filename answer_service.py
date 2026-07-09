"""Grounded answer generation with Claude.

Mirrors AnswerService.cs. The retrieved chunks are passed as numbered sources;
the system prompt confines Claude to those sources and requires citations —
that combination is what makes this RAG rather than "ask an LLM and hope".
"""

from anthropic import Anthropic

from search_index import ScoredChunk

# Triple-quoted strings are Python's multi-line strings (C# raw """...""").
SYSTEM_PROMPT = """\
You are a documentation assistant. Answer the user's question using ONLY the
numbered sources provided in the message. Cite sources inline as [1], [2], etc.
If the sources do not contain the answer, say so explicitly — do not invent facts.
"""


class AnswerService:
    def __init__(self) -> None:
        # Reads the ANTHROPIC_API_KEY environment variable automatically,
        # exactly like `new AnthropicClient()` in the C# SDK.
        self._client = Anthropic()

    def ask(self, question: str, context: list[ScoredChunk]) -> str:
        # Build the numbered-sources block. enumerate(..., start=1) gives
        # (index, item) pairs — the Python idiom for LINQ's .Select((c, i) => ...).
        sources = "\n\n".join(
            f"[{i}] (from {chunk.source_file})\n{chunk.content}"
            for i, chunk in enumerate(context, start=1)
        )

        response = self._client.messages.create(
            model="claude-opus-4-8",
            max_tokens=16000,
            # Adaptive thinking: Claude decides per-request whether and how
            # much to reason before answering.
            thinking={"type": "adaptive"},
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": f"Sources:\n\n{sources}\n\nQuestion: {question}",
                }
            ],
        )

        # The response content is a list of typed blocks (thinking blocks,
        # text blocks, ...). Keep only the text — same filtering the .NET
        # version does with .OfType<TextBlock>().
        return "\n".join(
            block.text for block in response.content if block.type == "text"
        )
