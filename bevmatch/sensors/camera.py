"""Camera modality: image-embedding place descriptor (§3.1 hloc/DBoW analogue).

A camera observation carries a global image embedding (from any backbone). The
descriptor compares embeddings by cosine distance — the same retrieval framework
as LiDAR, just a different representation (Principle 2). The embedding vector is
also a valid index-backend prefilter vector, so FAISS works unchanged.
"""

from __future__ import annotations

import numpy as np

from bevmatch.core.datamodel import Observation, Scene
from bevmatch.retrieval.base import DescriptorCode, GlobalDescriptor

MODALITY = "camera_embedding"


def camera_scene(scene_id: str, embedding: np.ndarray, place_id: str | None = None, **kw) -> Scene:
    """Build a Scene from a camera image embedding."""
    obs = Observation(MODALITY, points=np.zeros((0, 2)), embedding=np.asarray(embedding, dtype=float))
    return Scene(scene_id=scene_id, observations={MODALITY: obs}, place_id=place_id, **kw)


class CameraEmbeddingDescriptor(GlobalDescriptor):
    name = "camera-embedding"

    def extract(self, scene: Scene) -> DescriptorCode:
        emb = scene.primary().embedding
        if emb is None:
            raise ValueError("camera scene has no embedding")
        v = np.asarray(emb, dtype=float)
        return DescriptorCode(vector=v, payload=v)

    def distance(self, query: DescriptorCode, ref: DescriptorCode) -> tuple[float, float | None]:
        a, b = query.vector, ref.vector
        cos = float(a @ b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9)
        return 1.0 - cos, None
