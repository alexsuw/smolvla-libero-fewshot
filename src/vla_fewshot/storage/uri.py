"""Parse object-storage URIs without embedding bucket or host names."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse


class ObjectUriError(ValueError):
    """Raised when an object URI is missing or uses an unsupported scheme."""


@dataclass(frozen=True)
class ObjectLocation:
    scheme: str
    bucket: str | None
    prefix: str
    raw: str


def parse_object_uri(uri: str) -> ObjectLocation:
    """Accept ``s3://bucket/prefix`` or ``file:///absolute/prefix``."""

    text = uri.strip()
    if not text:
        raise ObjectUriError("object URI is empty")
    parsed = urlparse(text)
    scheme = parsed.scheme.lower()
    if scheme == "s3":
        bucket = parsed.netloc.strip()
        if not bucket:
            raise ObjectUriError("s3 URI must include a bucket name")
        prefix = unquote(parsed.path.lstrip("/"))
        return ObjectLocation(scheme="s3", bucket=bucket, prefix=prefix, raw=text)
    if scheme == "file":
        path = unquote(parsed.path)
        if parsed.netloc:
            path = f"/{parsed.netloc}{path}"
        if not path:
            raise ObjectUriError("file URI must include an absolute path")
        resolved = str(Path(path))
        return ObjectLocation(scheme="file", bucket=None, prefix=resolved, raw=text)
    raise ObjectUriError(
        f"unsupported object URI scheme {scheme!r}; expected s3:// or file://"
    )


def join_key(*parts: str) -> str:
    """Join object-key fragments without producing a leading slash."""

    chunks: list[str] = []
    for part in parts:
        cleaned = part.strip().strip("/")
        if cleaned:
            chunks.append(cleaned)
    return "/".join(chunks)
