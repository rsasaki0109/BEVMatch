"""Optional FAISS-backed index (§3.3, §7.2 Index Backend Plugin).

Drops in behind ``IndexBackend`` for city-scale databases where brute force is
too slow. Requires ``faiss`` (``pip install faiss-cpu``); imported lazily so the
core stays numpy-only.
"""

from __future__ import annotations

import numpy as np

from bevmatch.retrieval.index import IndexBackend, _normalize


class FaissIndex(IndexBackend):
    """Inner-product FAISS index over L2-normalised vectors (= cosine)."""

    name = "faiss"

    def __init__(self) -> None:
        self._index = None

    def build(self, vectors: np.ndarray) -> None:
        import faiss  # optional dependency

        vectors = _normalize(np.atleast_2d(np.asarray(vectors, dtype="float32")))
        self._index = faiss.IndexFlatIP(vectors.shape[1])
        self._index.add(vectors)

    def search(self, query: np.ndarray, k: int) -> list[int]:
        if self._index is None or self._index.ntotal == 0:
            return []
        q = _normalize(np.asarray(query, dtype="float32").reshape(1, -1))
        k = min(k, self._index.ntotal)
        _, idx = self._index.search(q, k)
        return [int(i) for i in idx[0] if i >= 0]
