"""Training records, tokenization with loss masks, and batching for the student trainer.

Contents
  * The single-turn record: a chat prompt (system + user messages) and one assistant response.
  * Record -> student token ids + per-token loss weight. Only the response and its end-of-turn
    token carry weight; prompt, system, template and padding tokens are asserted to be 0.
  * TPU v6e alignment checks (multiples of 256).
  * The global micro-batch iterator (every JAX process computes the same order).

Nothing in this file computes, rescales, clamps or offsets a metric.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import random
from typing import Any, Iterator, Sequence

import numpy as np

MXU = 256  # v6e systolic array edge (AGENTS.md)

# Segment codes stored next to every token so masks can be audited offline.
SEG = {
    "template": 0,  # chat-template glue (role headers, separators)
    "system": 1,
    "prompt": 2,  # prompt messages (system and user)
    "assistant": 3,  # the trained response
    "pad": 5,
}


def assert_mxu_multiple(name: str, value: int) -> None:
  if value <= 0 or value % MXU != 0:
    raise ValueError(f"{name}={value} must be a positive multiple of {MXU} (AGENTS.md)")


def sha256_file(path: str | os.PathLike[str]) -> str:
  h = hashlib.sha256()
  with open(path, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
      h.update(chunk)
  return h.hexdigest()


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Record:
  """One single-turn training example.

  JSONL layout: {"id", "messages": [{"role", "content"}, ...], "response", "db_id"?, "group_id"?,
  "reward"?}. `messages` is the prompt exactly as the model is served it (system and user
  messages, ending with a user message); `response` is the assistant text, trained verbatim.
  SFT rows have no reward; GRPO rows carry `group_id` (the question) and `reward`. Other keys
  (for example "sample" or "served_model") are provenance and are ignored here.
  """

  id: str
  messages: tuple[dict[str, str], ...]
  response: str
  db_id: str | None = None
  group_id: str | None = None
  reward: float | None = None

  @classmethod
  def from_dict(cls, d: dict[str, Any]) -> Record:
    rid, msgs, response = d["id"], d["messages"], d["response"]
    if not isinstance(rid, str) or not rid:
      raise ValueError(f"record id must be a non-empty string, got {rid!r}")
    if not isinstance(response, str):
      raise TypeError(f"{rid}: response must be a string")
    if not isinstance(msgs, list) or not msgs:
      raise ValueError(f"{rid}: messages must be a non-empty list")
    for m in msgs:
      if not isinstance(m, dict) or not isinstance(m.get("content"), str) or m.get("role") not in ("system", "user"):
        raise ValueError(f"{rid}: prompt messages must be system or user messages with string content (single turn)")
    if msgs[-1]["role"] != "user":
      raise ValueError(f"{rid}: messages must end with a user message")
    reward = d.get("reward")
    return cls(id=rid, messages=tuple({"role": m["role"], "content": m["content"]} for m in msgs), response=response,
               db_id=d.get("db_id"), group_id=d.get("group_id"), reward=None if reward is None else float(reward))


def load_records(path: str) -> list[Record]:
  out = []
  with open(path) as f:
    for line_no, line in enumerate(f, 1):
      line = line.strip()
      if not line:
        continue
      try:
        out.append(Record.from_dict(json.loads(line)))
      except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"{path}:{line_no}: not a training record ({e})") from e
  return out


# ---------------------------------------------------------------------------
# record -> token ids + loss weights (char-span based, template agnostic)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Msg:
  role: str
  content: str
  seg: str  # key of SEG
  weight: float  # 1.0 = train on this message's content tokens


def record_to_messages(rec: Record) -> list[Msg]:
  """The chat the student sees: every prompt message with weight 0, then the response with weight 1."""
  msgs = [Msg(m["role"], m["content"], "prompt", 0.0) for m in rec.messages]
  msgs.append(Msg("assistant", rec.response, "assistant", 1.0))
  return msgs


_IM_END = "<|im_end|>"
_THINK_START = "<think>"
_THINK_END = "</think>"
# Largest fraction of examples a builder may drop because the chat template rewrote a message
# beyond recognition; more than this means the renderer does not fit the template.
MAX_TEMPLATE_MISMATCH_FRACTION = 0.01


class TemplateMismatch(ValueError):
  """The rendered chat string does not contain a message's content."""


@dataclasses.dataclass
class Rendered:
  id: str
  input_ids: np.ndarray  # int32 [L_pad]
  loss_weight: np.ndarray  # float32 [L_pad]
  seg: np.ndarray  # uint8 [L_pad]
  n_real: int
  n_trained: int


def _char_spans(text: str, msgs: Sequence[Msg]) -> tuple[np.ndarray, np.ndarray]:
  """Per-character (segment code, weight) for a rendered chat string."""
  seg = np.full(len(text), SEG["template"], dtype=np.uint8)
  w = np.zeros(len(text), dtype=np.float32)
  cursor = 0
  for m in msgs:
    if not m.content:
      continue
    start = text.find(m.content, cursor)
    needle = m.content
    if start < 0:
      needle = m.content.strip()
      start = text.find(needle, cursor)
    end = start + len(needle)
    if start < 0 and m.role == "assistant" and _THINK_END in m.content:
      # Thinking templates split "reasoning</think>answer" and re-render it as
      # "<think>\nreasoning\n</think>\n\nanswer": locate both parts; the span between them
      # (the rendered "</think>") belongs to the response.
      reasoning = m.content.split(_THINK_END)[0].split(_THINK_START)[-1].strip()
      answer = m.content.split(_THINK_END)[-1].strip()
      start = text.find(reasoning, cursor) if reasoning else text.find(_THINK_END, cursor)
      tail = text.find(answer, start + len(reasoning)) if start >= 0 else -1
      if tail < 0 or _THINK_END not in text[start:tail]:
        start = -1
      else:
        end = tail + len(answer)
    if start < 0:
      raise TemplateMismatch(f"chat template altered a {m.seg} message; cannot locate its content")
    seg[start:end] = SEG[m.seg]
    if m.weight:
      w[start:end] = m.weight
      if m.role == "assistant" and text.startswith(_IM_END, end):
        # Train the end-of-turn token so the student learns to stop.
        seg[end : end + len(_IM_END)] = SEG["assistant"]
        w[end : end + len(_IM_END)] = m.weight
        end += len(_IM_END)
    cursor = end
  return seg, w


def render_example(rec: Record, tokenizer: Any, *, max_seq_len: int, overlong: str = "drop") -> Rendered | None:
  """Tokenize one record with the student's own tokenizer and chat template.

  Returns None when the example is longer than max_seq_len and overlong="drop"
  (the caller counts and reports drops; nothing is silently truncated).
  """
  assert_mxu_multiple("max_seq_len", max_seq_len)
  msgs = record_to_messages(rec)
  chat = [{"role": m.role, "content": m.content} for m in msgs]
  text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
  cseg, cw = _char_spans(text, msgs)
  enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
  ids = np.asarray(enc["input_ids"], dtype=np.int32)
  offs = enc["offset_mapping"]
  seg = np.empty(len(ids), dtype=np.uint8)
  w = np.empty(len(ids), dtype=np.float32)
  for t, (s, e) in enumerate(offs):
    e = max(e, s + 1)
    span_w = cw[s:e]
    span_seg = cseg[s:e]
    # A token is trained only if every character it covers is trained.
    w[t] = float(span_w.min()) if span_w.size else 0.0
    seg[t] = span_seg[0] if span_seg.size else SEG["template"]

  n_real = len(ids)
  if n_real > max_seq_len:
    if overlong == "drop":
      return None
    if overlong != "truncate":
      raise ValueError(f"overlong={overlong}")
    ids, w, seg = ids[:max_seq_len], w[:max_seq_len], seg[:max_seq_len]
    n_real = max_seq_len

  pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
  pad = max_seq_len - n_real
  ids = np.concatenate([ids, np.full(pad, pad_id, np.int32)])
  w = np.concatenate([w, np.zeros(pad, np.float32)])
  seg = np.concatenate([seg, np.full(pad, SEG["pad"], np.uint8)])
  assert_loss_mask(w, seg, rec.id)
  return Rendered(rec.id, ids, w, seg, n_real, int((w > 0).sum()))


def assert_loss_mask(weight: np.ndarray, seg: np.ndarray, rid: str) -> None:
  """AGENTS.md: prompt / system / template / pad tokens carry loss weight 0."""
  for name in ("prompt", "pad", "system", "template"):
    bad = weight[seg == SEG[name]]
    if bad.size and float(np.abs(bad).max()) != 0.0:
      raise AssertionError(f"{rid}: non-zero loss weight on {name} tokens")
  if weight.size % MXU:
    raise AssertionError(f"{rid}: sequence length {weight.size} not a multiple of {MXU}")


def build_sft_arrays(records: Sequence[Record], tokenizer: Any, *, max_seq_len: int,
                     overlong: str) -> tuple[list[Rendered], dict[str, Any]]:
  """Render every SFT record; drops (overlong, template mismatch) are counted and listed."""
  kept, dropped, mismatched = [], [], []
  for rec in records:
    try:
      r = render_example(rec, tokenizer, max_seq_len=max_seq_len, overlong=overlong)
    except TemplateMismatch:
      mismatched.append(rec.id)
      continue
    if r is None:
      dropped.append(rec.id)
    elif r.n_trained == 0:
      raise ValueError(f"{rec.id}: zero trained tokens after masking")
    else:
      kept.append(r)
  stats = {
      "n_input": len(records),
      "n_kept": len(kept),
      "n_dropped_overlong": len(dropped),
      "dropped_ids": dropped,
      "n_dropped_template_mismatch": len(mismatched),
      "template_mismatch_ids": mismatched,
      "tokens_real": int(sum(r.n_real for r in kept)),
      "tokens_trained": int(sum(r.n_trained for r in kept)),
  }
  if len(mismatched) > MAX_TEMPLATE_MISMATCH_FRACTION * max(len(records), 1):
    raise TemplateMismatch(f"{len(mismatched)} of {len(records)} examples do not fit the chat template: {mismatched[:5]}")
  return kept, stats


def bucket_length(n: int, buckets: Sequence[int]) -> int:
  """Smallest bucket >= n (buckets ascending)."""
  for b in buckets:
    if n <= b:
      return b
  raise ValueError(f"length {n} exceeds the largest bucket {buckets[-1]}")


def global_micro_batches(
    examples: Sequence[Rendered],
    *,
    global_batch: int,
    micro_batch: int,
    epochs: int,
    seed: int,
    buckets: Sequence[int] | None = None,
) -> Iterator[dict[str, np.ndarray]]:
  """Yield every global micro-batch in full (all rows), in a seeded per-epoch order.

  Every host holds the same examples and computes the same order, so each host can
  place the full micro-batch on the mesh (trainer_core._global) and devices keep only
  their own rows. Incomplete trailing global batches are dropped (counted by the caller).

  With `buckets`, the examples of each global batch are ordered by real length and each
  micro-batch is cut to the smallest bucket that holds its longest example. Padding is
  at the end of every example and carries weight 0, so only pad positions are removed;
  the global batch (and the token-weighted loss over it) is unchanged.
  """
  assert_mxu_multiple("global_batch", global_batch)
  if global_batch % micro_batch:
    raise ValueError("global_batch % micro_batch must be 0")
  if buckets:
    buckets = sorted(buckets)
    for b in buckets:
      assert_mxu_multiple("bucket", b)
  n_full = (len(examples) // global_batch) * global_batch
  for ep in range(epochs):
    order = list(range(len(examples)))
    random.Random(seed + ep).shuffle(order)
    order = order[:n_full]
    for gb_start in range(0, n_full, global_batch):
      gb = order[gb_start : gb_start + global_batch]
      if buckets:
        gb = sorted(gb, key=lambda i: (examples[i].n_real, i))
      for mb_start in range(0, global_batch, micro_batch):
        idx = gb[mb_start : mb_start + micro_batch]
        L = len(examples[idx[0]].input_ids)
        if buckets:
          L = bucket_length(max(examples[i].n_real for i in idx), buckets)
          for i in idx:
            if np.any(examples[i].seg[L:] != SEG["pad"]):
              raise AssertionError(f"example {examples[i].id}: non-pad tokens beyond bucket {L}")
        yield {
            "input_tokens": np.stack([examples[i].input_ids[:L] for i in idx]),
            "loss_weight": np.stack([examples[i].loss_weight[:L] for i in idx]),
            "pad_mask": np.stack([(examples[i].seg[:L] != SEG["pad"]).astype(np.int32) for i in idx]),
        }
