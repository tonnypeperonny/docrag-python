"""Paragraph-aware text chunking.

Paragraphs are accumulated up to a target size; the last paragraph of each
chunk is carried over into the next one ("overlap") so a fact that straddles
a chunk boundary is still retrievable from at least one chunk.
"""

from collections.abc import Iterator
from dataclasses import dataclass

TARGET_CHARS = 1200


@dataclass(frozen=True)
class Chunk:
    """One retrievable unit of text."""
    source_file: str
    ordinal: int  # position of this chunk within its source file
    content: str


def split(
    source_file: str,
    text: str,
    target_chars: int = TARGET_CHARS,
    overlap: bool = True,
) -> Iterator[Chunk]:
    """Split `text` into chunks of roughly `target_chars` characters.

    `target_chars` and `overlap` are exposed so ingestion runs can compare
    chunking strategies (small vs large chunks, with vs without overlap).
    """
    # A "paragraph" is anything separated by a blank line.
    paragraphs = [
        p.strip()
        for p in text.replace("\r\n", "\n").split("\n\n")
        if p.strip()
    ]

    buffer: list[str] = []
    buffer_len = 0
    ordinal = 0

    for paragraph in paragraphs:
        # If adding this paragraph would overflow the target size, emit the
        # current buffer as a finished chunk first.
        if buffer and buffer_len + len(paragraph) > target_chars:
            yield Chunk(source_file, ordinal, "\n\n".join(buffer))
            ordinal += 1

            if overlap:
                carry = buffer[-1]
                buffer = [carry]
                buffer_len = len(carry)
            else:
                buffer = []
                buffer_len = 0

        buffer.append(paragraph)
        buffer_len += len(paragraph)

    if buffer:
        yield Chunk(source_file, ordinal, "\n\n".join(buffer))
