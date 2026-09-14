"""Abstract storage interface.

All storage backends (MinIO, local, S3) implement this interface.
This allows swapping storage without changing business logic.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class StoredObject:
    key: str
    size: int
    last_modified: str


class BaseStorage(ABC):
    """File storage interface."""

    @abstractmethod
    def put(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
        """Store an object."""
        ...

    @abstractmethod
    def get(self, key: str) -> bytes:
        """Retrieve an object. Raises KeyError if not found."""
        ...

    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete an object. Raises KeyError if not found."""
        ...

    @abstractmethod
    def list_objects(self) -> list[StoredObject]:
        """List all objects in the store."""
        ...

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if an object exists."""
        ...
