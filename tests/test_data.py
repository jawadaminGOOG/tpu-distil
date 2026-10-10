"""trainer/data.py: the single-turn record, loss masks on a character tokenizer, and the batch iterator."""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "trainer"))
import data as dc  # noqa: E402


class _CharTokenizer:
  pad_token_id = 0
  eos_token_id = 0

  def apply_chat_template(self, chat, tokenize=False, add_generation_prompt=False):
    return "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in chat)

  def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
    return {"input_ids": [ord(c) + 1 for c in text], "offset_mapping": [(i, i + 1) for i in range(len(text))]}


MSGS = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "QUESTION"}]


def _rec(rid="train-1#s0", response="ANSWER", msgs=MSGS, **kw):
  return dc.Record.from_dict({"id": rid, "messages": msgs, "response": response, **kw})


def test_record_fields_and_extra_keys():
  r = _rec(db_id="shop", sample=3, served_model="t")
  assert (r.id, r.response, r.db_id, r.group_id, r.reward) == ("train-1#s0", "ANSWER", "shop", None, None)
  assert r.messages == tuple(MSGS)
  g = _rec(group_id="train-1", reward=1)
  assert g.group_id == "train-1" and g.reward == 1.0


@pytest.mark.parametrize("msgs", [
    [],
    [{"role": "system", "content": "S"}],                                                      # no user message
    [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}, {"role": "user", "content": "Q2"}],
    [{"role": "user", "content": "Q"}, {"role": "tool", "content": "out"}],
    [{"role": "user", "content": 3}],
])
def test_record_refuses_anything_but_a_single_turn_prompt(msgs):
  with pytest.raises(ValueError):
    _rec(msgs=msgs)


def test_load_records_reports_the_line(tmp_path):
  p = tmp_path / "r.jsonl"
  p.write_text(json.dumps({"id": "a", "messages": MSGS, "response": "x"}) + "\n\n" +
               json.dumps({"id": "b", "messages": MSGS}) + "\n")
  with pytest.raises(ValueError, match=r"r.jsonl:3: not a training record"):
    dc.load_records(str(p))


def test_only_the_response_and_end_of_turn_are_trained():
  r = dc.render_example(_rec(), _CharTokenizer(), max_seq_len=256)
  text = "".join(chr(int(t) - 1) for t in r.input_ids[: r.n_real])
  assert text == "<|im_start|>system\nSYS<|im_end|>\n<|im_start|>user\nQUESTION<|im_end|>\n<|im_start|>assistant\nANSWER<|im_end|>\n"
  trained = "".join(chr(int(t) - 1) for t, w in zip(r.input_ids, r.loss_weight) if w > 0)
  assert trained == "ANSWER<|im_end|>" and r.n_trained == len(trained)
  assert set(np.unique(r.loss_weight).tolist()) == {0.0, 1.0}
  assert np.all(r.loss_weight[r.seg == dc.SEG["prompt"]] == 0) and np.all(r.seg[r.n_real:] == dc.SEG["pad"])
  assert r.input_ids.shape == (256,) and np.all(r.input_ids[r.n_real:] == 0)


def test_loss_mask_assertion_and_alignment():
  w = np.zeros(256, np.float32)
  seg = np.full(256, dc.SEG["prompt"], np.uint8)
  dc.assert_loss_mask(w, seg, "x")
  w[3] = 1.0
  with pytest.raises(AssertionError, match="prompt"):
    dc.assert_loss_mask(w, seg, "x")
  with pytest.raises(AssertionError, match="multiple"):
    dc.assert_loss_mask(np.zeros(100, np.float32), np.zeros(100, np.uint8), "x")
  with pytest.raises(ValueError, match="multiple of 256"):
    dc.render_example(_rec(), _CharTokenizer(), max_seq_len=100)


def test_overlong_drop_and_truncate_and_stats():
  long = _rec(rid="long", response="A" * 300)
  kept, stats = dc.build_sft_arrays([_rec(), long], _CharTokenizer(), max_seq_len=256, overlong="drop")
  assert [r.id for r in kept] == ["train-1#s0"] and stats["dropped_ids"] == ["long"]
  t = dc.render_example(long, _CharTokenizer(), max_seq_len=256, overlong="truncate")
  assert t.n_real == 256 and t.n_trained > 0


def test_template_that_rewrites_content_is_counted_and_refused():
  class Rewriting(_CharTokenizer):
    def apply_chat_template(self, chat, **kw):
      return super().apply_chat_template([dict(m, content=m["content"].lower()) for m in chat], **kw)

  with pytest.raises(dc.TemplateMismatch):
    dc.build_sft_arrays([_rec()], Rewriting(), max_seq_len=256, overlong="drop")


def test_global_micro_batches_order_buckets_and_trailing_drop(monkeypatch):
  monkeypatch.setattr(dc, "MXU", 2)
  tok = _CharTokenizer()
  recs = [_rec(rid=f"r{i}", response="A" * (5 + 40 * (i % 3))) for i in range(7)]
  ex, _ = dc.build_sft_arrays(recs, tok, max_seq_len=256, overlong="drop")
  batches = list(dc.global_micro_batches(ex, global_batch=4, micro_batch=2, epochs=2, seed=3, buckets=[128, 256]))
  assert len(batches) == 2 * 2  # 7 examples -> 1 full global batch of 4 per epoch, 2 micro-batches each
  again = list(dc.global_micro_batches(ex, global_batch=4, micro_batch=2, epochs=2, seed=3, buckets=[128, 256]))
  assert all(np.array_equal(a["input_tokens"], b["input_tokens"]) for a, b in zip(batches, again))
  for b in batches:
    n = b["pad_mask"].sum(axis=1).max()
    assert b["input_tokens"].shape[1] == (128 if n <= 128 else 256)
    assert np.all(b["loss_weight"][b["pad_mask"] == 0] == 0)
  # within a global batch, micro-batches are ordered by real length
  lens = [int(b["pad_mask"].sum(axis=1).max()) for b in batches[:2]]
  assert lens == sorted(lens)
