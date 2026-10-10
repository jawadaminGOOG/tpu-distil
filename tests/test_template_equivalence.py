"""The trainer's single-turn training context is exactly the prompt vLLM's OpenAI server feeds the model.

For each real student tokenizer, a single-turn example (system + user -> one assistant response) is
rendered by trainer/data.py render_example from the record layout distil/build_sft.py writes. Checked:
  * context: the tokens before the first trained token are the token ids of
    apply_chat_template(messages, add_generation_prompt=True[, enable_thinking=False]), tokenized as
    vLLM tokenizes a chat prompt (no added special tokens), and decode to that exact string;
  * target: the trained tokens are tokenizer(response) + <|im_end|>, i.e. the text response + "<|im_end|>".
    The newline the template writes after <|im_end|> is template glue with weight 0 (not trained);
  * everything else (system, user, role headers, the empty think block, padding) has weight 0.
A negative control renders with a deliberately wrong template and shows the check fails.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "trainer"))
sys.path.insert(0, str(REPO / "distil"))

transformers = pytest.importorskip("transformers")
hub = pytest.importorskip("huggingface_hub")

import bird  # noqa: E402
import data as dc  # noqa: E402

# repo -> chat_template kwargs of the serving request (generate.py --chat-template-kwargs): Qwen3-1.7B
# is queried with thinking disabled; Qwen3-4B-Instruct-2507's template has no thinking mode.
STUDENTS = {
    "Qwen/Qwen3-1.7B": {"enable_thinking": False},
    "Qwen/Qwen3-4B-Instruct-2507": {},
}
EMPTY_THINK = "<think>\n\n</think>\n\n"

RESPONSES = [
    "The question asks for the number of customers in Oslo, so filter customer by city.\n"
    "```sql\nSELECT COUNT(*) FROM customer WHERE city = 'Oslo'\n```",
    "```sql\nSELECT name FROM customer ORDER BY name;\n```",
    "Join orders to customer on customer_id and sum the amounts.\n\n```sql\nSELECT c.name, SUM(o.amount)\n"
    "FROM customer AS c JOIN orders AS o ON o.customer_id = c.id\nGROUP BY c.name\n```\n",
    " SELECT 1",
]


@pytest.fixture(scope="module", params=sorted(STUDENTS))
def student(request):
  repo = request.param
  try:
    d = hub.snapshot_download(repo, allow_patterns=["*.json", "*.txt", "*.jinja"], local_files_only=True)
  except Exception:  # pylint: disable=broad-except
    pytest.skip(f"{repo} tokenizer files are not in the local Hugging Face cache")
  return repo, transformers.AutoTokenizer.from_pretrained(d)


def _messages():
  item = bird.Item(id="train-0", db_id="shop", question="How many customers live in Oslo?", evidence="",
                   gold_sql="", difficulty="simple")
  schema = "CREATE TABLE customer (id INTEGER PRIMARY KEY, name TEXT, city TEXT);"
  return bird.messages(item, schema)


def _row(response):
  """A row in the layout build_sft.py writes (messages = the full system + user chat)."""
  return {"id": "train-0#s0", "messages": _messages(), "response": response, "db_id": "shop", "sample": 0}


def _render(tok, response):
  row = _row(response)
  rec = dc.Record.from_dict(json.loads(json.dumps(row)))
  return dc.render_example(rec, tok, max_seq_len=2048), row


def check_equivalence(tok, chat_kwargs, rendered, row):
  """Raise AssertionError unless the rendered example has the serving context and the exact target."""
  msgs, response = row["messages"], row["response"]
  ids = rendered.input_ids[: rendered.n_real].tolist()
  w = rendered.loss_weight[: rendered.n_real]
  trained = np.flatnonzero(w > 0)
  assert trained.size, "no trained token"
  first, last = int(trained[0]), int(trained[-1])
  assert np.all(w[first : last + 1] == 1.0), "trained span is not contiguous"
  serving = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **chat_kwargs)
  assert tok.decode(ids[:first]) == serving, "training context string != serving prompt"
  assert ids[:first] == tok(serving, add_special_tokens=False)["input_ids"], "training context tokens != serving"
  im_end = tok.convert_tokens_to_ids("<|im_end|>")
  assert ids[first : last + 1] == tok(response, add_special_tokens=False)["input_ids"] + [im_end]
  assert tok.decode(ids[first : last + 1]) == response + "<|im_end|>"
  assert tok.decode(ids[last + 1 :]) == "\n"  # template glue after the turn, weight 0
  assert float(np.abs(rendered.loss_weight[last + 1 :]).sum()) == 0.0


@pytest.mark.parametrize("response", RESPONSES)
def test_training_context_is_the_serving_prompt(student, response):
  repo, tok = student
  rendered, row = _render(tok, response)
  check_equivalence(tok, STUDENTS[repo], rendered, row)


def test_qwen3_hybrid_template_context_holds_the_empty_think_block(student):
  """Qwen3-1.7B: the serving prompt ends with the empty think block, and so does the training context."""
  repo, tok = student
  serving = tok.apply_chat_template(_messages(), tokenize=False, add_generation_prompt=True, **STUDENTS[repo])
  if repo == "Qwen/Qwen3-1.7B":
    assert serving.endswith("<|im_start|>assistant\n" + EMPTY_THINK)
  else:
    assert serving.endswith("<|im_start|>assistant\n") and "<think>" not in serving


class _WrongTemplate:
  """The real tokenizer with a deliberately wrong training render: the empty think block is removed
  (1.7B) or inserted (4B-Instruct-2507)."""

  def __init__(self, tok, repo):
    self._tok, self._repo = tok, repo

  def __getattr__(self, name):
    return getattr(self._tok, name)

  def __call__(self, *a, **k):
    return self._tok(*a, **k)

  def apply_chat_template(self, chat, **kw):
    text = self._tok.apply_chat_template(chat, **kw)
    if self._repo == "Qwen/Qwen3-1.7B":
      return text.replace(EMPTY_THINK, "")
    return text.replace("<|im_start|>assistant\n", "<|im_start|>assistant\n" + EMPTY_THINK)


def test_negative_control_wrong_rendering_fails_the_check(student):
  repo, tok = student
  rendered, row = _render(_WrongTemplate(tok, repo), RESPONSES[0])
  with pytest.raises(AssertionError, match="serving"):
    check_equivalence(tok, STUDENTS[repo], rendered, row)


def test_prompt_messages_render_once(student):
  """build_sft rows carry the full chat in messages; each message appears exactly once."""
  _, tok = student
  rendered, row = _render(tok, RESPONSES[1])
  text = tok.decode(rendered.input_ids[: rendered.n_real].tolist())
  assert text.count(row["messages"][-1]["content"]) == 1 and text.count("<|im_start|>user") == 1
  assert text.count("<|im_start|>system") == 1


@pytest.mark.parametrize("response", ["\nSELECT 1", "\n\nReasoning.\n```sql\nSELECT 1\n```"])
def test_leading_newline_breaks_equivalence_so_the_builder_drops_it(student, response):
  """A response that starts with a newline is not reproducible from the serving prompt (the hybrid
  template strips it; elsewhere it merges with the template's newline into one token). build_sft.py
  drops such responses; this test documents why."""
  repo, tok = student
  rendered, row = _render(tok, response)
  with pytest.raises(AssertionError):
    check_equivalence(tok, STUDENTS[repo], rendered, row)
