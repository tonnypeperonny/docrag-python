"""Elasticsearch access: index lifecycle, bulk ingestion, hybrid retrieval."""

from dataclasses import dataclass

from elasticsearch import Elasticsearch, helpers

from chunker import Chunk
from embedding_service import DIMENSIONS, EmbeddingService

INDEX_NAME = "doc-chunks-py"  # separate index so the .NET version's data survives
RRF_K = 60                    # standard dampening constant for Reciprocal Rank Fusion

# Retrieval modes, selectable per query so the branches can be compared in
# isolation: does hybrid actually beat either branch alone on our docs?
MODES = ("hybrid", "bm25", "knn")


@dataclass(frozen=True)
class ScoredChunk:
    """A retrieved chunk plus its relevance score."""
    source_file: str
    ordinal: int
    content: str
    score: float


class SearchIndex:
    def __init__(self, client: Elasticsearch, embedder: EmbeddingService) -> None:
        self._client = client
        self._embedder = embedder

    def recreate(self) -> None:
        """Drop and re-create the index with the right mapping."""
        self._client.indices.delete(index=INDEX_NAME, ignore_unavailable=True)

        self._client.indices.create(
            index=INDEX_NAME,
            mappings={
                "properties": {
                    "source_file": {"type": "keyword"},   # exact-match only
                    "ordinal": {"type": "integer"},
                    "content": {"type": "text"},          # analyzed → BM25 searchable
                    "embedding": {
                        "type": "dense_vector",           # the vector-search field
                        "dims": DIMENSIONS,
                        "similarity": "cosine",
                    },
                }
            },
        )

    def index(self, chunks: list[tuple[Chunk, list[float]]]) -> None:
        """Bulk-index (chunk, embedding) pairs."""
        # "_id" makes ingestion idempotent: re-running ingest overwrites
        # instead of duplicating.
        helpers.bulk(
            self._client,
            (
                {
                    "_index": INDEX_NAME,
                    "_id": f"{chunk.source_file}#{chunk.ordinal}",
                    "source_file": chunk.source_file,
                    "ordinal": chunk.ordinal,
                    "content": chunk.content,
                    "embedding": embedding,
                }
                for chunk, embedding in chunks
            ),
        )
        # Make the documents searchable immediately (ES refreshes every ~1s
        # by default; a CLI wants it now).
        self._client.indices.refresh(index=INDEX_NAME)

    def list_all(self) -> list[dict]:
        """Return every stored chunk document, embeddings included.

        Returns the raw _source dicts because this is a debugging view — we
        want to see exactly what the index stores.
        """
        response = self._client.search(
            index=INDEX_NAME,
            size=1000,
            query={"match_all": {}},
        )
        docs = [hit["_source"] for hit in response["hits"]["hits"]]
        return sorted(docs, key=lambda d: (d["source_file"], d["ordinal"]))

    def _bm25(self, query: str, size: int) -> list[dict]:
        """Keyword branch — "match" runs BM25 scoring."""
        response = self._client.search(
            index=INDEX_NAME,
            size=size,
            query={"match": {"content": query}},
        )
        return response["hits"]["hits"]

    def _knn(self, query: str, size: int) -> list[dict]:
        """Vector branch — approximate nearest-neighbour over the embeddings.

        num_candidates > k trades speed for recall inside each shard.
        """
        response = self._client.search(
            index=INDEX_NAME,
            size=size,
            knn={
                "field": "embedding",
                "query_vector": self._embedder.embed(query),
                "k": size,
                "num_candidates": 100,
            },
        )
        return response["hits"]["hits"]

    def search(self, query: str, mode: str = "hybrid", top_n: int = 5) -> list[ScoredChunk]:
        """Retrieve top chunks using one of MODES.

        For "bm25" and "knn" the score is the branch's native score (BM25 /
        cosine similarity mapped by ES), so scores are NOT comparable across
        modes — only the ranking is.
        """
        if mode == "bm25":
            hits = self._bm25(query, top_n)
        elif mode == "knn":
            hits = self._knn(query, top_n)
        elif mode == "hybrid":
            return self.hybrid_search(query, top_n)
        else:
            raise ValueError(f"Unknown mode '{mode}', expected one of {MODES}")

        return [
            ScoredChunk(
                source_file=hit["_source"]["source_file"],
                ordinal=hit["_source"]["ordinal"],
                content=hit["_source"]["content"],
                score=hit["_score"],
            )
            for hit in hits
        ]

    def hybrid_search(self, query: str, top_n: int = 5) -> list[ScoredChunk]:
        """BM25 + kNN, fused with Reciprocal Rank Fusion.

        Why two searches? They fail differently:
        - BM25 (keyword) nails exact terms — IDs, acronyms, product names —
          but misses paraphrases ("course" vs "training").
        - kNN (vector) nails paraphrases and synonyms but can miss rare
          exact tokens that the embedding model never learned.
        Fusing by RANK (not score) sidesteps the fact that BM25 scores and
        cosine similarities live on completely different scales.
        """
        per_branch = 20
        bm25 = self._bm25(query, per_branch)
        knn = self._knn(query, per_branch)

        # Reciprocal Rank Fusion:
        # score(doc) = sum over each ranking of 1 / (RRF_K + rank)
        # A doc ranked #1 in both lists gets 2/(60+1); a doc ranked #1 in
        # one list and absent from the other gets 1/61 — so agreement wins.
        fused: dict[str, tuple[dict, float]] = {}   # doc_id -> (source doc, score)

        for ranking in (bm25, knn):
            for rank, hit in enumerate(ranking, start=1):
                doc_id = hit["_id"]
                contribution = 1.0 / (RRF_K + rank)
                if doc_id in fused:
                    doc, score = fused[doc_id]
                    fused[doc_id] = (doc, score + contribution)
                else:
                    fused[doc_id] = (hit["_source"], contribution)

        top = sorted(fused.values(), key=lambda pair: pair[1], reverse=True)[:top_n]

        return [
            ScoredChunk(
                source_file=doc["source_file"],
                ordinal=doc["ordinal"],
                content=doc["content"],
                score=score,
            )
            for doc, score in top
        ]
