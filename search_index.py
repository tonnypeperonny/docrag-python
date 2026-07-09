"""Elasticsearch access: index lifecycle, bulk ingestion, hybrid retrieval.

Mirrors SearchIndex.cs. The interesting difference from .NET:

- The .NET client (Elastic.Clients.Elasticsearch) is *strongly typed* — index
  mappings and queries are built with typed descriptors and lambdas, and the
  compiler catches a wrong field name.
- The Python client is *dict-based* — you write the raw Elasticsearch JSON
  API as Python dictionaries. Nothing is checked until Elasticsearch itself
  rejects the request. More flexible, less safe: a classic tradeoff you'll
  see everywhere in Python.
"""

from dataclasses import dataclass

from elasticsearch import Elasticsearch, helpers

from chunker import Chunk
from embedding_service import DIMENSIONS, EmbeddingService

INDEX_NAME = "doc-chunks-py"  # separate index so the .NET version's data survives
RRF_K = 60                    # standard dampening constant for Reciprocal Rank Fusion


@dataclass(frozen=True)
class ScoredChunk:
    """A retrieved chunk plus its fused relevance score."""
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
        # ignore_unavailable: don't error if the index doesn't exist yet —
        # like `DeleteAsync` guarded by an Exists check in the .NET version.
        self._client.indices.delete(index=INDEX_NAME, ignore_unavailable=True)

        # The mapping as a plain dict — compare with the typed
        # .Properties<DocChunk>(p => p.Text(...).DenseVector(...)) in C#.
        # This IS the JSON you'd send with curl; Python dicts map 1:1 to JSON.
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
        """Bulk-index (chunk, embedding) pairs.

        `helpers.bulk` is the Python client's convenience wrapper over the
        _bulk API — the counterpart of .NET's BulkAsync + IndexMany.
        """
        # A generator expression feeding bulk actions — each dict is one
        # document. "_id" makes ingestion idempotent (re-running ingest
        # overwrites instead of duplicating), same as op.Id(doc.Id) in C#.
        helpers.bulk(
            self._client,
            (
                {
                    "_index": INDEX_NAME,
                    "_id": f"{chunk.source_file}#{chunk.ordinal}",  # f-string = C# $"..."
                    "source_file": chunk.source_file,
                    "ordinal": chunk.ordinal,
                    "content": chunk.content,
                    "embedding": embedding,
                }
                for chunk, embedding in chunks  # tuple unpacking in the loop head
            ),
        )
        # Make the documents searchable immediately (ES refreshes every ~1s
        # by default; tests and CLIs want it now).
        self._client.indices.refresh(index=INDEX_NAME)

    def list_all(self) -> list[dict]:
        """Return every stored chunk document, embeddings included.

        match_all is Elasticsearch's "SELECT *". We return the raw _source
        dicts here (not dataclasses) because this is a debugging view —
        we want to see exactly what the index stores.
        """
        response = self._client.search(
            index=INDEX_NAME,
            size=1000,
            query={"match_all": {}},
        )
        docs = [hit["_source"] for hit in response["hits"]["hits"]]
        # Sort by (file, ordinal). A tuple as the sort key is the Python
        # idiom for OrderBy(...).ThenBy(...) in LINQ.
        return sorted(docs, key=lambda d: (d["source_file"], d["ordinal"]))

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
        query_vector = self._embedder.embed(query)

        # Branch 1 — classic keyword search ("match" runs BM25 scoring).
        bm25 = self._client.search(
            index=INDEX_NAME,
            size=per_branch,
            query={"match": {"content": query}},
        )

        # Branch 2 — approximate nearest-neighbour search over the vectors.
        # num_candidates > k trades speed for recall inside each shard.
        knn = self._client.search(
            index=INDEX_NAME,
            size=per_branch,
            knn={
                "field": "embedding",
                "query_vector": query_vector,
                "k": per_branch,
                "num_candidates": 100,
            },
        )

        # --- Reciprocal Rank Fusion -------------------------------------
        # score(doc) = sum over each ranking of 1 / (RRF_K + rank)
        # A doc ranked #1 in both lists gets 2/(60+1); a doc ranked #1 in
        # one list and absent from the other gets 1/61 — so agreement wins.
        fused: dict[str, tuple[dict, float]] = {}   # doc_id -> (source doc, score)

        for response in (bm25, knn):
            # The response is a dict mirroring ES JSON: hits.hits[]._source
            for rank, hit in enumerate(response["hits"]["hits"], start=1):
                doc_id = hit["_id"]
                contribution = 1.0 / (RRF_K + rank)
                if doc_id in fused:
                    doc, score = fused[doc_id]
                    fused[doc_id] = (doc, score + contribution)
                else:
                    fused[doc_id] = (hit["_source"], contribution)

        # sorted(..., key=lambda ...) is Python's OrderByDescending;
        # [:top_n] slices the first N items (like .Take(top_n)).
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
