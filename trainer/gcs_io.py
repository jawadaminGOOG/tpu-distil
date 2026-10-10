"""Minimal GCS I/O for the trainers (google-cloud-storage; auth = pod workload identity).

  ensure_local(uri, cache_root) -> local path. gs:// prefixes are downloaded once
      per host under cache_root (default $GCS_CACHE or /tmp/gcs_cache; a `.complete`
      marker makes re-runs skip); local paths pass through. With $GCS_CACHE_MIN_FREE_BYTES
      set, complete cache entries not used by this process are evicted, least recently
      used first, until the download leaves at least that much free space on the disk.
  upload_dir(local_dir, gs_uri)  -> number of files uploaded (call on process 0 only).
  upload_file(local_path, gs_uri)
"""

from __future__ import annotations

import concurrent.futures as cf
import os
import pathlib
import shutil
import sys

_USED: set[str] = set()  # cache entries this process has ensured; never evicted


def is_gcs(uri: str) -> bool:
  return str(uri).startswith("gs://")


def split_gcs(uri: str) -> tuple[str, str]:
  if not is_gcs(uri):
    raise ValueError(f"not a gs:// uri: {uri}")
  bucket, _, prefix = uri[len("gs://") :].partition("/")
  return bucket, prefix.strip("/")


def cache_entries(cache_root: str | os.PathLike[str]) -> list[tuple[pathlib.Path, int, float]]:
  """(entry dir, bytes, last use) of every complete cache entry (a dir holding `.complete`)."""
  out = []
  for marker in pathlib.Path(cache_root).rglob(".complete"):
    d = marker.parent
    size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
    out.append((d, size, marker.stat().st_mtime))
  return out


def plan_evictions(entries, *, free: int, need: int, reserve: int, keep: set[str]) -> list[pathlib.Path]:
  """Least recently used entries (not in `keep`) to delete so that free - need >= reserve."""
  plan, avail = [], free
  for d, size, _ in sorted(entries, key=lambda e: e[2]):
    if avail - need >= reserve:
      break
    if str(d) in keep:
      continue
    plan.append(d)
    avail += size
  if avail - need < reserve:
    raise OSError(f"cache cannot make room: need {need} bytes + reserve {reserve}, at most {avail} free after eviction")
  return plan


def _make_room(cache_root: str, need: int) -> None:
  reserve = int(os.environ.get("GCS_CACHE_MIN_FREE_BYTES", "0") or 0)
  if reserve <= 0:
    return
  pathlib.Path(cache_root).mkdir(parents=True, exist_ok=True)
  free = shutil.disk_usage(cache_root).free
  for d in plan_evictions(cache_entries(cache_root), free=free, need=need, reserve=reserve, keep=_USED):
    size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
    print(f"[gcs] evicting cache entry {d} ({size} bytes) to keep {reserve} bytes free", file=sys.stderr, flush=True)
    shutil.rmtree(d)


def _client():
  from google.cloud import storage  # pylint: disable=g-import-not-at-top

  return storage.Client()


def ensure_local(uri: str, cache_root: str | None = None, workers: int = 32) -> str:
  if not is_gcs(uri):
    return uri
  cache_root = cache_root or os.environ.get("GCS_CACHE", "/tmp/gcs_cache")
  bucket_name, prefix = split_gcs(uri)
  dst = pathlib.Path(cache_root) / bucket_name / prefix
  marker = dst / ".complete"
  if marker.exists():
    marker.touch()
    _USED.add(str(dst))
    return str(dst)
  client = _client()
  bucket = client.bucket(bucket_name)
  blob = bucket.blob(prefix)
  if prefix and blob.exists():  # single object
    dst.parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(str(dst))
    return str(dst)
  blobs = [b for b in client.list_blobs(bucket_name, prefix=prefix + "/") if not b.name.endswith("/")]
  if not blobs:
    raise FileNotFoundError(f"nothing under {uri}")
  if dst.exists():  # an earlier download stopped before writing the marker
    print(f"[gcs] removing incomplete download {dst}", file=sys.stderr, flush=True)
    shutil.rmtree(dst)
  _make_room(cache_root, sum(int(b.size or 0) for b in blobs))

  def fetch(b):
    out = dst / b.name[len(prefix) + 1 :]
    out.parent.mkdir(parents=True, exist_ok=True)
    b.download_to_filename(str(out))

  with cf.ThreadPoolExecutor(workers) as ex:
    list(ex.map(fetch, blobs))
  marker.write_text(f"{len(blobs)}\n")
  _USED.add(str(dst))
  print(f"[gcs] {uri} -> {dst} ({len(blobs)} objects)", file=sys.stderr, flush=True)
  return str(dst)


def upload_file(local_path: str, gs_uri: str) -> None:
  bucket_name, name = split_gcs(gs_uri)
  _client().bucket(bucket_name).blob(name).upload_from_filename(local_path)


def upload_dir(local_dir: str, gs_uri: str, workers: int = 32) -> int:
  bucket_name, prefix = split_gcs(gs_uri)
  bucket = _client().bucket(bucket_name)
  root = pathlib.Path(local_dir)
  files = [p for p in root.rglob("*") if p.is_file() and p.name != ".complete"]

  def put(p: pathlib.Path):
    bucket.blob(f"{prefix}/{p.relative_to(root).as_posix()}").upload_from_filename(str(p))

  with cf.ThreadPoolExecutor(workers) as ex:
    list(ex.map(put, files))
  print(f"[gcs] {local_dir} -> {gs_uri} ({len(files)} files)", file=sys.stderr, flush=True)
  return len(files)


def local_or_tmp(uri: str, tmp_root: str = "/tmp/tpu_logs") -> str:
  """Where to write an output named by `uri` before upload (identity for local paths)."""
  if not is_gcs(uri):
    return uri
  bucket, prefix = split_gcs(uri)
  return os.path.join(tmp_root, bucket, prefix)
