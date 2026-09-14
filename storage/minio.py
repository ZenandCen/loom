"""MinIO (S3-compatible) storage implementation."""

import logging
import os

import boto3

from storage.base import BaseStorage, StoredObject

logger = logging.getLogger(__name__)


class MinIOStorage(BaseStorage):
    """MinIO/S3 storage backend."""

    def __init__(
        self,
        endpoint: str | None = None,
        access_key: str | None = None,
        secret_key: str | None = None,
        bucket: str | None = None,
        use_ssl: bool | None = None,
    ):
        self.endpoint = endpoint or os.getenv("MINIO_ENDPOINT", "")
        self.access_key = access_key or os.getenv("MINIO_ACCESS_KEY", "")
        self.secret_key = secret_key or os.getenv("MINIO_SECRET_KEY", "")
        self.bucket = bucket or os.getenv("MINIO_BUCKET", "")
        self.use_ssl = use_ssl if use_ssl is not None else os.getenv("MINIO_USE_SSL", "false").lower() == "true"

        if not all([self.endpoint, self.access_key, self.secret_key, self.bucket]):
            raise ValueError(
                "MinIO config missing. Set MINIO_ENDPOINT, MINIO_ACCESS_KEY, "
                "MINIO_SECRET_KEY, MINIO_BUCKET in .env"
            )

        self.client = boto3.client(
            "s3",
            endpoint_url=f"{'https' if self.use_ssl else 'http'}://{self.endpoint}",
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
        )
        self._ensure_bucket()

    def _ensure_bucket(self):
        try:
            self.client.head_bucket(Bucket=self.bucket)
        except self.client.exceptions.ClientError:
            self.client.create_bucket(Bucket=self.bucket)
            logger.info(f"Created bucket: {self.bucket}")

    def put(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
        self.client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=data,
            ContentType=content_type,
        )
        logger.info(f"PUT {self.bucket}/{key} ({len(data)} bytes)")

    def get(self, key: str) -> bytes:
        resp = self.client.get_object(Bucket=self.bucket, Key=key)
        return resp["Body"].read()

    def delete(self, key: str) -> None:
        self.client.delete_object(Bucket=self.bucket, Key=key)
        logger.info(f"DELETE {self.bucket}/{key}")

    def list_objects(self) -> list[StoredObject]:
        resp = self.client.list_objects_v2(Bucket=self.bucket)
        return [
            StoredObject(
                key=obj["Key"],
                size=obj["Size"],
                last_modified=obj["LastModified"].isoformat(),
            )
            for obj in resp.get("Contents", [])
        ]

    def exists(self, key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except self.client.exceptions.ClientError:
            return False
