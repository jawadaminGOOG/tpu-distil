"""Chat-completions client for the teacher endpoint: retries with backoff and bounded concurrency.

The endpoint is an OpenAI-compatible base URL (for example a local `vertex_openai_proxy.py`).
Transient failures (HTTP 408/429/5xx, a scaled-to-zero endpoint that is still starting, network
errors) are retried with exponential backoff; any other HTTP error is raised at once. The default
retry budget (about 55 minutes) covers a cold start of a scaled-to-zero GPU endpoint.
"""
from __future__ import annotations

import concurrent.futures
import json
import re
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Iterable

RETRY_HTTP = frozenset({408, 429, 500, 502, 503, 504})
RETRY_TEXT = ("not yet ready", "scale-up", "temporarily unavailable")
THINK_END = "</think>"
_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


class TeacherError(RuntimeError):
  """A teacher call that failed after all retries, or returned no usable answer."""


def _retryable(code: int, text: str) -> bool:
  low = text.lower()
  return code in RETRY_HTTP or any(t in low for t in RETRY_TEXT)


def call(endpoint: str, body: dict[str, Any], *, api_key: str = "sk-local", timeout: float = 900,
         max_attempts: int = 16, first_delay: float = 5.0, max_delay: float = 300.0,
         sleep: Callable[[float], None] = time.sleep,
         opener: Callable[..., Any] = urllib.request.urlopen) -> dict[str, Any]:
  """POST `body` to <endpoint>/chat/completions and return the decoded JSON response."""
  url = endpoint.rstrip("/") + "/chat/completions"
  data = json.dumps(body).encode("utf-8")
  delay = first_delay
  last = ""
  for attempt in range(1, max_attempts + 1):
    req = urllib.request.Request(url, data=data, method="POST", headers={
        "Content-Type": "application/json", "Authorization": f"Bearer {api_key}"})
    try:
      with opener(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
      text = e.read().decode("utf-8", "replace")[:2000]
      last = f"HTTP {e.code}: {text}"
      if not _retryable(e.code, text):
        raise TeacherError(last) from e
    except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as e:
      last = f"{type(e).__name__}: {e}"
    if attempt < max_attempts:
      sleep(min(delay, max_delay))
      delay *= 2
  raise TeacherError(f"gave up after {max_attempts} attempts: {last}")


def content(resp: dict[str, Any]) -> str | None:
  """Raw message content of a chat-completions response, or None."""
  try:
    c = resp["choices"][0]["message"]["content"]
  except (KeyError, IndexError, TypeError):
    return None
  return c if isinstance(c, str) else None


def answer(resp: dict[str, Any]) -> str | None:
  """Message content with any reasoning (text up to the last `</think>`) removed."""
  c = content(resp)
  if c is None:
    return None
  return c.rsplit(THINK_END, 1)[1].strip() if THINK_END in c else c.strip()


def finish_reason(resp: dict[str, Any]) -> Any:
  try:
    return resp["choices"][0].get("finish_reason")
  except (KeyError, IndexError, TypeError, AttributeError):
    return None


def answer_json(resp: dict[str, Any]) -> dict[str, Any]:
  """The answer parsed as one JSON object (a surrounding code fence is allowed)."""
  text = answer(resp)
  if not text:
    raise TeacherError(f"empty answer (finish_reason={finish_reason(resp)!r})")
  m = _FENCE.match(text)
  if m:
    text = m.group(1)
  try:
    obj = json.loads(text)
  except json.JSONDecodeError:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
      raise TeacherError(f"answer is not JSON: {text[:200]!r}") from None
    try:
      obj = json.loads(text[start:end + 1])
    except json.JSONDecodeError as e:
      raise TeacherError(f"answer is not JSON: {text[:200]!r}") from e
  if not isinstance(obj, dict):
    raise TeacherError(f"answer is not a JSON object: {text[:200]!r}")
  return obj


def map_ordered(fn: Callable[[Any], Any], items: Iterable[Any], workers: int = 8) -> list[Any]:
  """[fn(x) for x in items] on a thread pool; an exception is returned in place of its result."""
  items = list(items)
  out: list[Any] = [None] * len(items)
  with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
    futs = {ex.submit(fn, x): i for i, x in enumerate(items)}
    for f in concurrent.futures.as_completed(futs):
      i = futs[f]
      try:
        out[i] = f.result()
      except Exception as e:  # pylint: disable=broad-except
        out[i] = e
  return out
