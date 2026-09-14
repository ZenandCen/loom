"""Storage layer — file persistence backends."""

from storage.base import BaseStorage, StoredObject
from storage.minio import MinIOStorage

__all__ = ["BaseStorage", "StoredObject", "MinIOStorage"]
