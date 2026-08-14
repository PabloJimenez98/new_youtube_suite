from pathlib import Path

import pytest

from app.storage import LocalStorageBackend, StorageKeyNotFoundError


def test_save_get_delete_round_trip(tmp_path: Path) -> None:
    backend = LocalStorageBackend(tmp_path)
    key = LocalStorageBackend.make_key("episode-01.mp4")

    returned_key = backend.save(key, b"fake video bytes")
    assert returned_key == key

    with backend.get(key) as f:
        assert f.read() == b"fake video bytes"

    backend.delete(key)
    with pytest.raises(StorageKeyNotFoundError):
        backend.get(key)


def test_get_missing_key_raises(tmp_path: Path) -> None:
    backend = LocalStorageBackend(tmp_path)
    with pytest.raises(StorageKeyNotFoundError):
        backend.get("does-not-exist")


def test_delete_missing_key_raises(tmp_path: Path) -> None:
    backend = LocalStorageBackend(tmp_path)
    with pytest.raises(StorageKeyNotFoundError):
        backend.delete("does-not-exist")


def test_make_key_strips_path_components() -> None:
    key = LocalStorageBackend.make_key("../../etc/passwd")
    assert "/" not in key
    assert key.endswith("_passwd")


def test_save_creates_base_path_if_missing(tmp_path: Path) -> None:
    nested = tmp_path / "does" / "not" / "exist" / "yet"
    backend = LocalStorageBackend(nested)
    assert nested.is_dir()
