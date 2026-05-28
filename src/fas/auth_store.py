from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class FaceTemplateStore:
    def __init__(self, store_path: str | Path = "data/face_templates.json") -> None:
        self.store_path = Path(store_path)
        self.store_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.store_path.exists():
            self.store_path.write_text("{}", encoding="utf-8")

    def _load(self) -> dict[str, Any]:
        try:
            return json.loads(self.store_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    def _save(self, data: dict[str, Any]) -> None:
        self.store_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def save_template(self, user_id: str, embedding: np.ndarray) -> None:
        data = self._load()
        data[user_id] = {
            "embedding": embedding.astype(float).tolist(),
        }
        self._save(data)

    def get_template(self, user_id: str) -> np.ndarray | None:
        data = self._load()
        user_data = data.get(user_id)

        if not user_data:
            return None

        embedding = user_data.get("embedding")
        if not embedding:
            return None

        return np.asarray(embedding, dtype=np.float32)

    def user_exists(self, user_id: str) -> bool:
        data = self._load()
        return user_id in data