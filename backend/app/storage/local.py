import io
import uuid
from pathlib import Path
from typing import BinaryIO

from app.storage.base import StorageBackend, StorageKeyNotFoundError


class LocalStorageBackend(StorageBackend):
    """Stores files on the local filesystem under `base_path`."""

    def __init__(self, base_path: Path | str) -> None:
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def make_key(original_filename: str) -> str:
        """Generate a collision-safe, path-traversal-safe storage key.

        Never derive the on-disk path directly from a client-supplied
        filename: it could contain `../` segments or collide with an
        existing file.
        """
        safe_name = Path(original_filename).name  # strip any directory components
        return f"{uuid.uuid4()}_{safe_name}"

    def _resolve(self, key: str) -> Path:
        # `Path(key).name` guards against a key smuggling `../` back in.
        return self.base_path / Path(key).name

    def path_for(self, key: str) -> Path:
        """Resolve `key` to an absolute filesystem path.

        Local-backend-only (not part of the abstract `StorageBackend`
        interface) -- used where a real path is genuinely needed, e.g.
        `FileResponse` for Range-request-capable file streaming.
        """
        return self._resolve(key)

    def save(self, key: str, data: BinaryIO | bytes) -> str:
        path = self._resolve(key)
        if isinstance(data, bytes):
            data = io.BytesIO(data)
        with path.open("wb") as f:
            f.write(data.read())
        return key

    def get(self, key: str) -> BinaryIO:
        path = self._resolve(key)
        if not path.is_file():
            raise StorageKeyNotFoundError(key)
        return path.open("rb")

    def delete(self, key: str) -> None:
        path = self._resolve(key)
        if not path.is_file():
            raise StorageKeyNotFoundError(key)
        path.unlink()
