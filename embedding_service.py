"""Sentence embeddings — and the biggest .NET-vs-Python difference in this project.

In the .NET version (EmbeddingService.cs) we had to hand-roll the whole
pipeline: load a WordPiece tokenizer, run the ONNX transformer ourselves,
mean-pool the hidden states over the attention mask, then L2-normalize.
~80 lines of tensor code.

In Python, the `sentence-transformers` library does ALL of that in one call.
This is the real story of Python in ML: the ecosystem, not the language.
The model is the same (all-MiniLM-L6-v2, 384 dimensions), so the two versions
produce equivalent vectors.
"""

from sentence_transformers import SentenceTransformer

DIMENSIONS = 384  # output vector size of all-MiniLM-L6-v2


class EmbeddingService:
    """Thin wrapper so main.py reads the same as the .NET Program.cs."""

    def __init__(self) -> None:
        # __init__ is the constructor. `self` is explicit in Python —
        # it's what `this` is in C#, but you must declare and use it.
        #
        # First run downloads the model (~90 MB) from HuggingFace into
        # ~/.cache/huggingface. Subsequent runs load it from that cache —
        # no manual download script needed (the .NET version needed one).
        self._model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        # The leading underscore in `_model` is a *convention* for "private".
        # Python has no real access modifiers — no `private` keyword.

    def embed(self, text: str) -> list[float]:
        """Text in, 384-dim unit vector out.

        One line replaces the .NET version's tokenize / forward-pass /
        mean-pool / normalize code. `normalize_embeddings=True` gives us
        the L2 normalization we did by hand in C#.
        """
        vector = self._model.encode(text, normalize_embeddings=True)
        # encode() returns a numpy array; the Elasticsearch client wants
        # plain Python floats, so convert. numpy is Python's tensor
        # workhorse — the equivalent of what DenseTensor<float> was in .NET.
        return vector.tolist()
