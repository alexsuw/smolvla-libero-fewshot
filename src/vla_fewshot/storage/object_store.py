"""Object-store backends. The public API cannot delete remote objects."""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Protocol

from vla_fewshot.storage.uri import ObjectLocation, ObjectUriError, parse_object_uri


class ObjectStoreError(RuntimeError):
    """Raised when a remote object operation fails closed."""


class ObjectStore(Protocol):
    def exists(self, key: str) -> bool:
        """Return whether ``key`` is present."""

    def size(self, key: str) -> int:
        """Return object size in bytes."""

    def get_bytes(self, key: str) -> bytes:
        """Download the full object."""

    def put_bytes(self, key: str, payload: bytes) -> None:
        """Upload bytes. Overwriting a different payload is the caller's decision."""


class FileObjectStore:
    """Local filesystem backend used for protocol tests and dry-run evidence."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def _path(self, key: str) -> Path:
        relative = key.strip("/")
        if not relative or relative.startswith("..") or "/../" in f"/{relative}/":
            raise ObjectStoreError(f"refusing unsafe object key: {key!r}")
        return self.root / relative

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    def size(self, key: str) -> int:
        path = self._path(key)
        if not path.is_file():
            raise FileNotFoundError(key)
        return path.stat().st_size

    def get_bytes(self, key: str) -> bytes:
        path = self._path(key)
        if not path.is_file():
            raise FileNotFoundError(key)
        return path.read_bytes()

    def put_bytes(self, key: str, payload: bytes) -> None:
        _atomic_write_bytes(self._path(key), payload)


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


class S3ObjectStore:
    """boto3-backed S3 store. Imported only when an s3:// URI is executed."""

    def __init__(self, bucket: str, prefix: str) -> None:
        try:
            import boto3
        except ImportError as error:
            raise ObjectStoreError(
                "s3:// execute requires boto3 in the runtime environment; "
                "dry-run still works without it, or use file:// for protocol tests"
            ) from error
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self._client = boto3.client("s3")

    def _key(self, key: str) -> str:
        relative = key.strip("/")
        if self.prefix:
            return f"{self.prefix}/{relative}" if relative else self.prefix
        return relative

    def exists(self, key: str) -> bool:
        from botocore.exceptions import ClientError

        try:
            self._client.head_object(Bucket=self.bucket, Key=self._key(key))
        except ClientError as error:
            code = str(error.response.get("Error", {}).get("Code", ""))
            if code in {"404", "NoSuchKey", "NotFound"}:
                return False
            raise ObjectStoreError(f"s3 head failed for {key}: {error}") from error
        return True

    def size(self, key: str) -> int:
        from botocore.exceptions import ClientError

        try:
            response = self._client.head_object(Bucket=self.bucket, Key=self._key(key))
        except ClientError as error:
            raise FileNotFoundError(key) from error
        return int(response["ContentLength"])

    def get_bytes(self, key: str) -> bytes:
        from botocore.exceptions import ClientError

        try:
            response = self._client.get_object(Bucket=self.bucket, Key=self._key(key))
        except ClientError as error:
            raise FileNotFoundError(key) from error
        return response["Body"].read()

    def put_bytes(self, key: str, payload: bytes) -> None:
        self._client.put_object(Bucket=self.bucket, Key=self._key(key), Body=payload)


def open_object_store(uri: str) -> ObjectStore:
    location = parse_object_uri(uri)
    return store_from_location(location)


def store_from_location(location: ObjectLocation) -> ObjectStore:
    if location.scheme == "file":
        return FileObjectStore(Path(location.prefix))
    if location.scheme == "s3":
        if location.bucket is None:
            raise ObjectUriError("s3 URI is missing a bucket")
        return S3ObjectStore(location.bucket, location.prefix)
    raise ObjectUriError(f"unsupported scheme {location.scheme!r}")
