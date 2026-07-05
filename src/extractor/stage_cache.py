from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


CACHE_IDENTITY_SCHEMA_VERSION = 1
DEFAULT_CACHE_MODE = "strict"
CACHE_MODE_CHOICES = ("strict", "query-only", "off")


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


# Filesystem-location fields — absolute paths, OS filenames, file stems — must
# never enter a cache fingerprint. The cache is content-addressed, so the same
# inputs reused from a new location (e.g. after the source-centric layout pulls a
# master to <project>/media/source.mp4, renaming it) must yield the same key and
# reuse the cached response. These fields describe *where* a file lives, not
# *what* it contains, so they are stripped before a manifest/transcript is
# rendered into a prompt.
LOCATION_KEYS = ("path", "filename", "stem")


def without_location(value: dict[str, Any] | None) -> dict[str, Any]:
    """A copy of a metadata dict with filesystem-location keys removed, so it can
    be embedded in a cache fingerprint without coupling the key to where files
    live. See LOCATION_KEYS."""
    return {k: v for k, v in (value or {}).items() if k not in LOCATION_KEYS}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def normalize_cache_mode(value: str | None) -> str:
    mode = value or DEFAULT_CACHE_MODE
    if mode not in CACHE_MODE_CHOICES:
        raise ValueError(
            f"Unknown cache mode {mode!r}; expected one of {', '.join(CACHE_MODE_CHOICES)}."
        )
    return mode


def source_file_fingerprint(paths: Iterable[Path]) -> dict[str, Any]:
    root = repo_root()
    seen: set[Path] = set()
    files: list[dict[str, Any]] = []
    for raw_path in paths:
        path = raw_path.expanduser().resolve()
        if path in seen:
            continue
        seen.add(path)
        if not path.exists():
            files.append(
                {
                    "path": str(path),
                    "relative_path": str(path),
                    "exists": False,
                    "sha256": None,
                }
            )
            continue
        try:
            relative_path = str(path.relative_to(root))
        except ValueError:
            relative_path = str(path)
        files.append(
            {
                "path": str(path),
                "relative_path": relative_path,
                "exists": True,
                "sha256": sha256_file(path),
            }
        )

    files.sort(key=lambda item: item["relative_path"])
    hash_files = [
        {
            "relative_path": item["relative_path"],
            "exists": item["exists"],
            "sha256": item["sha256"],
        }
        for item in files
    ]
    return {
        "hash": sha256_text(stable_json(hash_files)),
        "files": files,
    }


def build_cache_identity(
    *,
    query_fingerprint_payload: dict[str, Any],
    source_files: Iterable[Path],
    cache_mode: str | None,
) -> dict[str, Any]:
    mode = normalize_cache_mode(cache_mode)
    query_hash = sha256_text(stable_json(query_fingerprint_payload))
    implementation = source_file_fingerprint(source_files)
    if mode == "query-only":
        cache_key = query_hash
    else:
        key_material: dict[str, Any] = {
            "cache_identity_schema_version": CACHE_IDENTITY_SCHEMA_VERSION,
            "mode": mode,
            "query_hash": query_hash,
        }
        key_material["implementation_hash"] = implementation["hash"]
        cache_key = sha256_text(stable_json(key_material))

    return {
        "schema_version": CACHE_IDENTITY_SCHEMA_VERSION,
        "mode": mode,
        "cache_key": cache_key,
        "query_hash": query_hash,
        "implementation_hash": implementation["hash"],
        "source_files": implementation["files"],
        "query_fingerprint_payload": query_fingerprint_payload,
    }


def request_cache_key(fingerprint_payload: dict[str, Any]) -> str:
    return sha256_text(stable_json(fingerprint_payload))


def cache_entry_dir(cache_dir: Path, cache_key: str) -> Path:
    return cache_dir / cache_key[:2] / cache_key


def cache_entry_exists(entry_dir: Path) -> bool:
    return (entry_dir / "raw-response.json").exists()


def cache_manifest_payload(
    *,
    cache_key: str,
    query_fingerprint_payload: dict[str, Any],
    source: str,
    cache_identity: dict[str, Any] | None,
    created_at: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 2,
        "created_at": created_at,
        "cache_key": cache_key,
        "source": source,
        "query_fingerprint_payload": query_fingerprint_payload,
        # Kept for compatibility with older cache readers and inspection scripts.
        "fingerprint_payload": query_fingerprint_payload,
    }
    if cache_identity is not None:
        payload["cache_identity"] = cache_identity
    return payload


def response_cache_status_payload(
    *,
    enabled: bool,
    hit: bool,
    cache_identity: dict[str, Any],
    entry_dir: Path | None,
) -> dict[str, Any]:
    return {
        "enabled": enabled,
        "hit": hit,
        "mode": cache_identity["mode"],
        "key": cache_identity["cache_key"] if enabled else None,
        "entry_dir": str(entry_dir) if entry_dir else None,
        "query_hash": cache_identity["query_hash"],
        "implementation_hash": cache_identity["implementation_hash"],
        "source_files": cache_identity["source_files"],
    }
