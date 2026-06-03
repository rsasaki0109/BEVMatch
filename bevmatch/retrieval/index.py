"""Index backend abstraction (§7.2 Index Backend Plugin).

v0.1/v0.2 ships a numpy brute-force index. A FAISS-backed index (Faiss is an
efficient dense-vector similarity search library, §3.3) can drop in behind the
same interface once the database outgrows brute force.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, 1e-9, None)


class IndexBackend(ABC):
    """Maps descriptor vectors to a searchable index returning row indices."""

    name: str = "index"

    @abstractmethod
    def build(self, vectors: np.ndarray) -> None:
        ...

    @abstractmethod
    def search(self, query: np.ndarray, k: int) -> list[int]:
        """Return up to ``k`` row indices, nearest first."""


class BruteForceIndex(IndexBackend):
    """Exact nearest-neighbour search over normalised vectors (cosine)."""

    name = "brute-force"

    def __init__(self) -> None:
        self._db: np.ndarray | None = None

    def build(self, vectors: np.ndarray) -> None:
        vectors = np.atleast_2d(np.asarray(vectors, dtype=float))
        self._db = _normalize(vectors)

    def search(self, query: np.ndarray, k: int) -> list[int]:
        if self._db is None or len(self._db) == 0:
            return []
        q = _normalize(np.asarray(query, dtype=float).reshape(1, -1))[0]
        sims = self._db @ q  # cosine similarity
        k = min(k, len(sims))
        # argpartition for the top-k, then sort those by similarity descending
        top = np.argpartition(-sims, k - 1)[:k]
        return list(top[np.argsort(-sims[top])])


def make_index(name: str = "brute-force") -> IndexBackend:
    if name == "brute-force":
        return BruteForceIndex()
    if name == "faiss":
        from bevmatch.retrieval.faiss_index import FaissIndex  # optional dep

        return FaissIndex()
    raise ValueError(f"Unknown index backend: {name!r}")
