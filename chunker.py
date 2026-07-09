"""Paragraph-aware text chunking.

Mirrors the .NET version's Chunker.cs. Same algorithm: accumulate paragraphs
up to a target size, and carry the last paragraph of each chunk over into the
next one ("overlap") so a fact that straddles a chunk boundary is still
retrievable from at least one chunk.
"""

# `dataclass` is Python's answer to C# records: it auto-generates __init__,
# __repr__, and __eq__ from the field declarations below.
# `frozen=True` makes instances immutable — like a C# `record` (init-only).
from dataclasses import dataclass

# Type hints. Python doesn't *enforce* types at runtime (it's dynamically
# typed), but hints document intent and let tools like mypy/pyright catch
# errors — think of them as optional compiler checks.
from collections.abc import Iterator

TARGET_CHARS = 1200  # module-level "constant" — by convention, UPPER_CASE


@dataclass(frozen=True)
class Chunk:
    """One retrievable unit of text. C# equivalent:
    `record Chunk(string SourceFile, int Ordinal, string Content)`."""
    source_file: str   # Python naming convention is snake_case, not PascalCase
    ordinal: int       # position of this chunk within its source file
    content: str


def split(source_file: str, text: str) -> Iterator[Chunk]:
    """Split `text` into overlapping chunks.

    This is a *generator* function — the Python equivalent of a C# iterator
    method with `yield return`. It produces chunks lazily, one at a time,
    instead of building a full list in memory.
    """
    # Normalize Windows line endings, then split on blank lines.
    # A "paragraph" is anything separated by an empty line.
    # This chained list comprehension is the Python idiom for LINQ's
    # .Split().Select(p => p.Trim()).Where(p => p.Length > 0)
    paragraphs = [
        p.strip()                                  # trim whitespace
        for p in text.replace("\r\n", "\n").split("\n\n")
        if p.strip()                               # drop empty paragraphs
    ]

    buffer: list[str] = []   # paragraphs accumulated for the current chunk
    buffer_len = 0
    ordinal = 0

    for paragraph in paragraphs:
        # If adding this paragraph would overflow the target size,
        # emit the current buffer as a finished chunk first.
        if buffer and buffer_len + len(paragraph) > TARGET_CHARS:
            # "\n\n".join(list) is Python's string.Join("\n\n", list)
            yield Chunk(source_file, ordinal, "\n\n".join(buffer))
            ordinal += 1

            # Overlap: start the next chunk with the LAST paragraph of the
            # previous one. buffer[-1] is Python's negative indexing —
            # same as buffer[^1] in modern C#.
            overlap = buffer[-1]
            buffer = [overlap]
            buffer_len = len(overlap)

        buffer.append(paragraph)
        buffer_len += len(paragraph)

    # Flush whatever is left after the loop.
    if buffer:
        yield Chunk(source_file, ordinal, "\n\n".join(buffer))
