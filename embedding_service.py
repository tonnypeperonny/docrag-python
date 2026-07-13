"""Sentence embeddings via sentence-transformers.

Same model as the .NET version (all-MiniLM-L6-v2, 384 dimensions), so the
two versions produce equivalent vectors.
"""

from sentence_transformers import SentenceTransformer

DIMENSIONS = 384  # output vector size of all-MiniLM-L6-v2


class EmbeddingService:
    def __init__(self) -> None:
        # First run downloads the model (~90 MB) into ~/.cache/huggingface;
        # subsequent runs load it from that cache.
        self._model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def embed(self, text: str) -> list[float]:
        """Text in, 384-dim unit vector out (L2-normalized, cosine-ready)."""
        vector = self._model.encode(text, normalize_embeddings=True)
        # numpy array -> plain floats; the Elasticsearch client wants the latter.
        return vector.tolist()
