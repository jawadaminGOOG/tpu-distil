"""FinQA: numerical reasoning over one page of an earnings report (text and a table).

Questions come from the FinQA release files `<root>/<split>.json` (splits train, dev and test).
Each question is shown with its report page: the text before the table, the table as pipe-separated
rows (header first) and the text after it. The model is asked to end with `Answer: <number>`, and with
`Answer: yes` or `Answer: no` only when the question is a yes/no question. The gold answer is the executed answer of the gold program
(`qa.exe_ans`): a number, or "yes"/"no".

Grading is pure computation (`score`): a numeric prediction p is correct when any of p, p/100 and
p*100 is within 1% of the gold value g (|x - g| <= 0.01 |g|), so a percentage may be written either
as a percent number or as a fraction. When |g| < 0.5, the unscaled p alone may also be within 0.005
absolute; the percent forms p/100 and p*100 never use the absolute bound. A prediction of 0 is
never correct for a non-zero gold. Both bounds carry a 1e-9 relative allowance for floating-point
rounding. yes/no answers match exactly, ignoring case.

The question group is the report page (`filename`, e.g. "V/2008/page_17.pdf"): training questions
on a page that also appears in the evaluation split are left out by prompts.py
(`--exclude-groups-from`) and refused by the builders (`--forbid-groups`).
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import re
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

NAME = "finqa"
LABEL = "FinQA"
SPLITS = ("train", "dev", "test")
EVAL_SPLIT = "test"
FULL_COVERAGE_SPLITS = ("dev", "test")
GROUP_KEY = "report"
GROUP_SHORT = "report"
GROUP_NOUN = "report"
PRED_KEY = "pred_answer"
ANSWER_SHORT = "answer"
ANSWER_NOUN = "answer"

REL_TOL = 0.01
ABS_TOL = 0.005
SMALL_GOLD = 0.5  # below this |gold|, the absolute tolerance applies
_FP_SLACK = 1 + 1e-9
YES_NO = ("yes", "no")
EMPTY_SECTION = "(none)"

PROMPT_VERSION = "finqa-2"
SYSTEM_PROMPT = "You are an expert financial analyst."
USER_TEMPLATE = (
    "Report text before the table:\n{pre_text}\n\n"
    "Table (one row per line, cells separated by \" | \", header row first):\n{table}\n\n"
    "Report text after the table:\n{post_text}\n\n"
    "Question: {question}\n\n"
    "Reason step by step, briefly. Then give the final answer on the last line in the form "
    "`Answer: <number>`: a plain number, no units or currency symbols; use a percentage number for "
    "percentages. Only if the question is a yes/no question, write `Answer: yes` or `Answer: no` instead."
)


def prompt_template_sha256() -> str:
  return hashlib.sha256((PROMPT_VERSION + "\x00" + SYSTEM_PROMPT + "\x00" + USER_TEMPLATE).encode()).hexdigest()


_REPORT_NAME = re.compile(r"^[A-Za-z0-9.\-]+/\d{4}/page_\d+\.pdf$")


def valid_group(name: str) -> bool:
  """Whether `name` has the form of a FinQA report page, e.g. "ABC/2009/page_49.pdf"."""
  return bool(_REPORT_NAME.match(name))


@dataclass(frozen=True)
class Item:
  id: str
  report: str
  question: str
  gold: float | str
  pre_text: tuple[str, ...]
  table: tuple[tuple[str, ...], ...]
  post_text: tuple[str, ...]
  source_id: str = ""

  @property
  def group(self) -> str:
    return self.report


def parse_gold(value) -> float | str:
  """The gold answer as a float, or "yes"/"no"."""
  if isinstance(value, bool):
    raise ValueError(f"gold answer {value!r} is not a number or yes/no")
  if isinstance(value, (int, float)):
    g = float(value)
  else:
    s = str(value).strip().lower()
    if s in YES_NO:
      return s
    g = float(s)
  if not math.isfinite(g):
    raise ValueError(f"gold answer {value!r} is not finite")
  return g


def load_items(root: Path, split: str) -> list[Item]:
  """Questions of `split` from `<root>/<split>.json`, numbered by position: id "<split>-<n>"."""
  rows = json.loads((Path(root) / f"{split}.json").read_text())
  items = []
  for n, r in enumerate(rows):
    qa = r["qa"]
    try:
      gold = parse_gold(qa["exe_ans"])
    except ValueError as e:
      raise ValueError(f"{split}.json row {n} ({r.get('id')}): {e}") from None
    items.append(Item(id=f"{split}-{n}", report=r["filename"], question=" ".join(qa["question"].split()),
                      gold=gold, pre_text=tuple(r.get("pre_text") or ()),
                      table=tuple(tuple(str(c) for c in row) for row in (r.get("table") or ())),
                      post_text=tuple(r.get("post_text") or ()), source_id=str(r.get("id", ""))))
  return items


def _text(lines: tuple[str, ...]) -> str:
  body = "\n".join(s.strip() for s in lines if s.strip())
  return body or EMPTY_SECTION


def table_text(table: tuple[tuple[str, ...], ...]) -> str:
  body = "\n".join(" | ".join(c.strip() for c in row) for row in table)
  return body or EMPTY_SECTION


def messages(item: Item, context: None = None) -> list[dict[str, str]]:
  """The chat for one question. The report page travels with the item, so `context` is unused."""
  del context
  user = USER_TEMPLATE.format(pre_text=_text(item.pre_text), table=table_text(item.table),
                              post_text=_text(item.post_text), question=item.question)
  return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


_QUESTION = re.compile(r"\n\nQuestion: (.*)\n\nReason step by step, briefly\.", re.DOTALL)


def question_of(user_text: str) -> str:
  """The question of a user message written by `messages` ("" if it is not in that form)."""
  head, sep, tail = user_text.rpartition("\n\nQuestion: ")
  if not sep:
    return ""
  m = _QUESTION.match("\n\nQuestion: " + tail)
  return m.group(1).strip() if m else ""


# ---------------------------------------------------------------------------
# Answer extraction and scoring
# ---------------------------------------------------------------------------

_ANSWER = re.compile(r"\banswer\s*:", re.IGNORECASE)
_NUM = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_NUMBER = re.compile(_NUM)
_TOKEN = re.compile(r"\b(yes|no)\b|(" + _NUM + ")", re.IGNORECASE)
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
_DECORATION = " \t*_`:#>"
_ACCOUNTING_NEGATIVE = re.compile(r"\(\s*(\d+(?:\.\d*)?|\.\d+)\s*%?\s*\)")
_SPACED_MINUS = re.compile(r"(?<![\w.)])-\s+(?=\.?\d)")


def _strip_number_marks(text: str) -> str:
  """Unicode minus to "-", currency "$" removed, thousands separators removed."""
  return _THOUSANDS.sub("", text.replace("\u2212", "-").replace("$", ""))


def _canonical_number(s: str) -> str:
  s = s.lstrip("+")
  return s[:-1] if s.endswith(".") else s


def _first_answer(text: str) -> str | None:
  text = _strip_number_marks(text)
  text = _SPACED_MINUS.sub("-", _ACCOUNTING_NEGATIVE.sub(r"-\1", text))
  m = _TOKEN.search(text)
  if not m:
    return None
  return m.group(1).lower() if m.group(1) else _canonical_number(m.group(2))


def extract_answer(text: str | None) -> str | None:
  """The final answer of a response: "yes", "no", or a plain number string ("12.5", "-3", "1234").

  The answer is read from the text after the last "Answer:" (any case) on its line; when that
  line holds nothing but markup, from the next non-empty line. The first "yes", "no" or number
  there is the answer; "$", "%", thousands separators and words around it are dropped, and an
  accounting negative "(3.2)" or a spaced minus "- 3.2" reads as -3.2. A response
  with no "Answer:" falls back to its last number. Text before a closing "</think>" is ignored.
  None when no answer can be read.
  """
  if not text:
    return None
  if "</think>" in text:
    text = text.rsplit("</think>", 1)[1]
  marks = list(_ANSWER.finditer(text))
  if marks:
    line, _, after = text[marks[-1].end():].partition("\n")
    if not line.strip(_DECORATION):
      line = next((ln for ln in after.splitlines() if ln.strip(_DECORATION)), "")
    return _first_answer(line)
  numbers = _NUMBER.findall(_strip_number_marks(text))
  return _canonical_number(numbers[-1]) if numbers else None


def normalize_answer(answer: str | None) -> str:
  """Numbers in a fixed float format ("12.50" and "12.5" are the same answer), yes/no in lower case."""
  if answer is None:
    return ""
  s = str(answer).strip().lower()
  try:
    x = float(s)
  except ValueError:
    return s
  return format(x, ".10g") if math.isfinite(x) else s


def _within_relative(x: float, g: float) -> bool:
  return abs(x - g) <= REL_TOL * abs(g) * _FP_SLACK


def _within_absolute(x: float, g: float) -> bool:
  return abs(g) < SMALL_GOLD and abs(x - g) <= ABS_TOL * _FP_SLACK


def score(pred: str | None, gold: float | str) -> tuple[bool, str]:
  """(correct, error) for one extracted answer. Errors: "no_answer" (nothing extracted),
  "not_a_number" (non-numeric answer to a numeric question), "not_yes_no"; "" when the answer was
  read and is simply wrong or right."""
  if pred is None or not str(pred).strip():
    return False, "no_answer"
  p = str(pred).strip().lower()
  if isinstance(gold, str):
    if p not in YES_NO:
      return False, "not_yes_no"
    return p == gold, ""
  try:
    x = float(p)
  except ValueError:
    return False, "not_a_number"
  if not math.isfinite(x):
    return False, "not_a_number"
  g = float(gold)
  if x == 0 and g != 0:
    return False, ""
  if _within_relative(x, g) or _within_absolute(x, g):
    return True, ""
  return any(_within_relative(c, g) for c in (x / 100, x * 100)), ""


def format_gold(gold: float | str) -> str:
  """The gold answer written the way a model writes it: plain decimal notation with thousands
  separators and every significant digit of the float (so a non-zero gold never reads as 0)."""
  if isinstance(gold, str):
    return gold
  s = format(Decimal(repr(float(gold))), ",f")
  if "." in s:
    s = s.rstrip("0").rstrip(".")
  return "0" if s in ("-0", "") else s


def gold_response(item: Item) -> str:
  return f"The answer follows from the report.\nAnswer: {format_gold(item.gold)}"


# ---------------------------------------------------------------------------
# Pipeline entry points (prompts.py, grade.py)
# ---------------------------------------------------------------------------


def prompt_rows(root: Path, split: str, items: list[Item]) -> tuple[list[dict], dict]:
  """One prompt row per item and the task-specific manifest fields (report pages, prompt sizes)."""
  del root
  rows = [{"id": it.id, "report": it.report, "split": split, "source_id": it.source_id, "messages": messages(it)}
          for it in items]
  chars = [len(r["messages"][1]["content"]) for r in rows]
  extra = {"reports": sorted({it.report for it in items}), "prompt_version": PROMPT_VERSION,
           "user_chars": {"max": max(chars, default=0), "mean": round(sum(chars) / max(len(chars), 1), 1)}}
  return rows, extra


def control_samples(items: dict[str, Item], kind: str, seed: int = 0) -> dict[str, list[tuple[int, str | None]]]:
  """Evaluator controls, each passed through `extract_answer` as a model response would be.

  gold: each question's gold answer, written as a model would write it (`gold_response`), graded
  against itself; must score 100%. shuffled: each question gets the gold answer of another question
  of the split with a different normalised gold answer, drawn with random.Random(seed); a negative
  control that must score low.
  """
  if kind == "gold":
    return {i: [(0, extract_answer(gold_response(it)))] for i, it in items.items()}
  if kind != "shuffled":
    raise ValueError(f"unknown control {kind!r}")
  rng = random.Random(seed)
  pool = list(items.values())
  norm = {it.id: normalize_answer(extract_answer(gold_response(it))) for it in pool}
  out: dict[str, list[tuple[int, str | None]]] = {}
  for it in pool:
    other = None
    for _ in range(64):
      cand = rng.choice(pool)
      if norm[cand.id] != norm[it.id]:
        other = cand
        break
    if other is None:
      rest = [o for o in pool if norm[o.id] != norm[it.id]]
      if not rest:
        continue
      other = rng.choice(rest)
    out[it.id] = [(0, extract_answer(gold_response(other)))]
  return out


def grade(root: Path, split: str, items: dict[str, Item], samples: dict[str, list[tuple[int, str | None]]], *,
          workers: int = 0, timeout: float = 0.0) -> list[dict]:
  """Graded rows {id, report, sample, pred_answer, correct, error}; scoring needs no data files."""
  del root, split, workers, timeout
  rows = []
  for qid, preds in samples.items():
    it = items[qid]
    for idx, pred in preds:
      ok, err = score(pred, it.gold)
      rows.append({"id": qid, "report": it.report, "sample": idx, "pred_answer": pred, "correct": ok, "error": err})
  return rows
