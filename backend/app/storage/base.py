from abc import ABC, abstractmethod
from typing import BinaryIO


class StorageKeyNotFoundError(Exception):
    """Raised when `get`/`delete` is called with a key that doesn't exist."""


class StorageBackend(ABC):
    """Thin interface over "wherever uploaded files actually live".

    v1 ships a single local-filesystem implementation
    (`LocalStorageBackend`); swapping to S3-compatible storage later
    means adding a new implementation of this interface, not touching
    any calling code. `storage_path` values are backend-relative keys,
    never absolute paths or URLs.
    """

    @abstractmethod
    def save(self, key: str, data: BinaryIO | bytes) -> str:
        """Persist `data` under `key`, returning the key actually used."""

    @abstractmethod
    def get(self, key: str) -> BinaryIO:
        """Return a readable binary stream for `key`.

        Raises `StorageKeyNotFoundError` if `key` doesn't exist.
        """

    @abstractmethod
    def delete(self, key: str) -> None:
        """Remove `key`.

        Raises `StorageKeyNotFoundError` if `key` doesn't exist.
        """
