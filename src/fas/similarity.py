from __future__ import annotations

import numpy as np


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return embedding
    return embedding / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = normalize_embedding(a)
    b = normalize_embedding(b)
    return float(np.dot(a, b))