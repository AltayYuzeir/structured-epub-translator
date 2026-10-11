from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from threading import Lock


class TranslationCache:
    """
    File-backed JSON cache.

    Cache keys include source text, language pair, model name and prompt
    version, so changing the prompt/model never silently reuses old output.
    """

    def __init__(self, root: str | Path = "cache"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    @staticmethod
    def key(
        source_text: str,
        source_language: str,
        target_language: str,
        model_name: str,
        prompt_version: str = "epub-segmented-v1",
    ) -> str:
        payload = "\n".join([
            prompt_version,
            model_name,
            source_language,
            target_language,
            source_text,
        ])
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _path(self, key: str) -> Path:
        return self.root / f"{key}.json"

    def get(self, key: str) -> str | None:
        path = self._path(key)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))["translation"]
        except (OSError, KeyError, json.JSONDecodeError):
            return None

    def put(self, key: str, translation: str, metadata: dict | None = None) -> None:
        path = self._path(key)
        payload = {
            "translation": translation,
            "metadata": metadata or {},
        }

        with self._lock:
            fd, tmp_name = tempfile.mkstemp(
                prefix=".cache-",
                suffix=".json",
                dir=str(self.root),
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)
                os.replace(tmp_name, path)
            finally:
                if os.path.exists(tmp_name):
                    os.unlink(tmp_name)
