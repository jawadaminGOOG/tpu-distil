"""teacher_client: retries transient failures, raises on others, parses JSON answers after reasoning."""
from __future__ import annotations

import io
import json
import pathlib
import sys
import urllib.error

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "distil"))
import teacher_client as tc  # noqa: E402


class _Resp(io.BytesIO):
  def __enter__(self):
    return self

  def __exit__(self, *a):
    return False


def _http_error(code, text):
  return urllib.error.HTTPError("u", code, "m", {}, io.BytesIO(text.encode()))


def test_teacher_client_retries_transient_then_succeeds():
  seq = [_http_error(400, "Model is not yet ready for inference."), _http_error(503, "busy"),
         _Resp(json.dumps({"choices": [{"message": {"content": "r</think>{\"a\": 1}"}}]}).encode())]
  sleeps = []

  def opener(req, timeout):
    x = seq.pop(0)
    if isinstance(x, Exception):
      raise x
    return x
  resp = tc.call("http://x/v1", {}, opener=opener, sleep=sleeps.append, first_delay=5.0)
  assert tc.answer_json(resp) == {"a": 1} and not seq and sleeps == [5.0, 10.0]


def test_teacher_client_raises_on_non_retryable():
  def opener(req, timeout):
    raise _http_error(400, "bad request")
  with pytest.raises(tc.TeacherError, match="HTTP 400"):
    tc.call("http://x/v1", {}, opener=opener, sleep=lambda s: None)


def test_teacher_client_gives_up_after_max_attempts():
  def opener(req, timeout):
    raise _http_error(503, "busy")
  with pytest.raises(tc.TeacherError, match="gave up after 3 attempts"):
    tc.call("http://x/v1", {}, opener=opener, sleep=lambda s: None, max_attempts=3)


def test_answer_strips_reasoning_and_map_ordered_keeps_order():
  assert tc.answer({"choices": [{"message": {"content": "think</think>  final "}}]}) == "final"
  assert tc.answer({"choices": []}) is None
  assert tc.answer_json({"choices": [{"message": {"content": "```json\n{\"b\": 2}\n```"}}]}) == {"b": 2}
  out = tc.map_ordered(lambda x: 1 / x, [1, 0, 2], workers=3)
  assert out[0] == 1.0 and isinstance(out[1], ZeroDivisionError) and out[2] == 0.5
