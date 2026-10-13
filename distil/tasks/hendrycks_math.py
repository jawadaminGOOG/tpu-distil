"""MATH: competition mathematics problems (Hendrycks et al.), seven subjects, levels 1-5.

Questions come from `<root>/train.jsonl` and `<root>/test.jsonl`, one problem per line with the
fields `source`, `subject`, `problem`, `level`, `type` and `solution` (the release's parquet files,
one row per problem). The dev split is not in the release: it is `DEV_FRACTION` of the train
problems (500 of 7,500), held out with random.Random(DEV_SEED), stratified by subject
(largest-remainder allocation); the train split is the rest. Two train problems have an empty boxed
answer; no response can match them.
Ids number the problems of each split in file order: "<split>-<n>".

The model is asked to put its final answer in `\\boxed{...}`. The gold answer is the content of the
last `\\boxed{...}` (or `\\fbox{...}`) in the reference solution. Grading is exact match after the
reference MATH normalisation (`strip_string`, from the benchmark's `math_equivalence.py`): it removes
spacing, `\\left`/`\\right`, degree and dollar signs, percent signs and trailing `\\text{ unit}`,
rewrites `tfrac`/`dfrac` as `frac`, `\\frac12` as `\\frac{1}{2}`, `a/b` as `\\frac{a}{b}` for integers,
`0.5` as `\\frac{1}{2}`, and drops a short "x =" prefix. Answers that are mathematically equal but
written differently (e.g. `2\\sqrt{2}` and `\\sqrt{8}`) do not match; the rule is the same for every model.

The question group is a fingerprint of the problem text (`problem_hash`, "p" + 16 hex digits of the
SHA-256 of the case/space-normalised problem), so a training problem identical to an evaluation
problem shares its group and is left out by prompts.py (`--exclude-groups-from`) and refused by the
builders (`--forbid-groups`). Near-duplicates are caught by the 13-gram filter (`overlap.py`).
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

NAME = "math"
LABEL = "MATH"
SPLITS = ("train", "dev", "test")
EVAL_SPLIT = "test"
FULL_COVERAGE_SPLITS = ("dev", "test")
GROUP_KEY = "problem_hash"
GROUP_SHORT = "problem"
GROUP_NOUN = "problem"
PRED_KEY = "pred_answer"
ANSWER_SHORT = "answer"
ANSWER_NOUN = "answer"

DEV_FRACTION = 1 / 15
DEV_SEED = 0

PROMPT_VERSION = "math-1"
SYSTEM_PROMPT = "You are an expert mathematician."
USER_TEMPLATE = (
    "Problem: {problem}\n\n"
    "Reason step by step, briefly. Then give the final answer on the last line in the form "
    "`\\boxed{{<answer>}}`, simplified, in LaTeX."
)


def prompt_template_sha256() -> str:
  return hashlib.sha256((PROMPT_VERSION + "\x00" + SYSTEM_PROMPT + "\x00" + USER_TEMPLATE).encode()).hexdigest()


_GROUP_NAME = re.compile(r"^p[0-9a-f]{16}$")


def valid_group(name: str) -> bool:
  """Whether `name` has the form of a problem fingerprint, e.g. "p0123456789abcdef"."""
  return bool(_GROUP_NAME.match(name))


def problem_hash(problem: str) -> str:
  norm = " ".join(re.findall(r"[a-z0-9]+", problem.lower()))
  return "p" + hashlib.sha256(norm.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class Item:
  id: str
  problem_hash: str
  question: str
  gold: str
  subject: str
  level: str
  source: str = ""

  @property
  def group(self) -> str:
    return self.problem_hash


def _read_jsonl(path: Path) -> list[dict]:
  with open(path) as f:
    return [json.loads(line) for line in f if line.strip()]


def dev_indices(rows: list[dict], fraction: float = DEV_FRACTION, seed: int = DEV_SEED) -> set[int]:
  """Positions of the train-file rows held out as dev: round(fraction * rows) rows, stratified by subject.

  Each subject gets its proportional share rounded down, and the remaining slots go to the subjects
  with the largest fractional remainders (ties by subject name). Within a subject the rows are drawn
  with one random.Random(seed), visiting subjects in sorted order.
  """
  by_subject: dict[str, list[int]] = {}
  for i, r in enumerate(rows):
    by_subject.setdefault(r["subject"], []).append(i)
  total = len(rows)
  size = round(fraction * total)
  quota = {s: size * len(ix) / total for s, ix in by_subject.items()}
  alloc = {s: int(q) for s, q in quota.items()}
  for s in sorted(quota, key=lambda s: (-(quota[s] - alloc[s]), s))[:size - sum(alloc.values())]:
    alloc[s] += 1
  rng = random.Random(seed)
  out: set[int] = set()
  for s in sorted(by_subject):
    out.update(rng.sample(by_subject[s], alloc[s]))
  return out


def load_items(root: Path, split: str) -> list[Item]:
  """Problems of `split`. dev and train both come from train.jsonl (see `dev_indices`)."""
  if split not in SPLITS:
    raise ValueError(f"unknown MATH split {split!r}")
  rows = _read_jsonl(Path(root) / ("test.jsonl" if split == "test" else "train.jsonl"))
  if split != "test":
    held = dev_indices(rows)
    rows = [r for i, r in enumerate(rows) if (i in held) == (split == "dev")]
  items = []
  for n, r in enumerate(rows):
    gold = extract_answer(r["solution"])
    if gold is None:
      raise ValueError(f"{split} row {n} ({r.get('source')}): the solution has no \\boxed answer")
    items.append(Item(id=f"{split}-{n}", problem_hash=problem_hash(r["problem"]), question=r["problem"].strip(),
                      gold=gold, subject=r["subject"], level=r.get("level", ""), source=r.get("source", "")))
  return items


def messages(item: Item, context: None = None) -> list[dict[str, str]]:
  """The chat for one problem. The problem carries everything, so `context` is unused."""
  del context
  return [{"role": "system", "content": SYSTEM_PROMPT},
          {"role": "user", "content": USER_TEMPLATE.format(problem=item.question)}]


def question_of(user_text: str) -> str:
  """The problem of a user message written by `messages` ("" if it is not in that form)."""
  head, sep, _ = user_text.rpartition("\n\nReason step by step, briefly.")
  if not sep or not head.startswith("Problem: "):
    return ""
  return head[len("Problem: "):].strip()


# ---------------------------------------------------------------------------
# Answer extraction and normalisation
# ---------------------------------------------------------------------------


def last_boxed(text: str) -> str | None:
  """The last `\\boxed{...}` / `\\fbox{...}` group of `text` (braces balanced), or `\\boxed x` up to
  the next "$"; None when there is none or its braces never close."""
  i = max(text.rfind("\\boxed"), text.rfind("\\fbox"))
  if i < 0:
    return None
  if text.startswith("\\boxed ", i):
    return text[i:].split("$")[0]
  depth, j = 0, i
  while j < len(text):
    if text[j] == "{":
      depth += 1
    elif text[j] == "}":
      depth -= 1
      if depth == 0:
        return text[i:j + 1]
    j += 1
  return None


def _unbox(s: str) -> str | None:
  if s.startswith("\\boxed "):
    return s[len("\\boxed "):]
  for left in ("\\boxed{", "\\fbox{"):
    if s.startswith(left) and s.endswith("}"):
      return s[len(left):-1]
  return None


def extract_answer(text: str | None) -> str | None:
  """The content of the last `\\boxed{...}` in a response, or None. Text before a closing
  "</think>" is ignored."""
  if not text:
    return None
  if "</think>" in text:
    text = text.rsplit("</think>", 1)[1]
  box = last_boxed(text)
  return None if box is None else _unbox(box)


def _fix_fracs(string: str) -> str:
  substrs = string.split("\\frac")
  new_str = substrs[0]
  for substr in substrs[1:]:
    new_str += "\\frac"
    if substr and substr[0] == "{":
      new_str += substr
      continue
    if len(substr) < 2:
      return string
    a, b = substr[0], substr[1]
    if b != "{":
      new_str += "{" + a + "}{" + b + "}" + substr[2:]
    else:
      new_str += "{" + a + "}" + b + substr[2:]
  return new_str


def _fix_a_slash_b(string: str) -> str:
  parts = string.split("/")
  if len(parts) != 2:
    return string
  try:
    a, b = int(parts[0]), int(parts[1])
  except ValueError:
    return string
  if string != f"{a}/{b}":
    return string
  return "\\frac{" + str(a) + "}{" + str(b) + "}"


class _Unnormalisable(ValueError):
  pass


def _remove_right_units(string: str) -> str:
  # "\text{ " (with the space) only describes units in MATH answers.
  if "\\text{ " in string:
    splits = string.split("\\text{ ")
    if len(splits) != 2:
      raise _Unnormalisable(string)
    return splits[0]
  return string


def _fix_sqrt(string: str) -> str:
  if "\\sqrt" not in string:
    return string
  splits = string.split("\\sqrt")
  new_string = splits[0]
  for split in splits[1:]:
    if not split:
      raise _Unnormalisable(string)
    new_string += ("\\sqrt{" + split[0] + "}" + split[1:]) if split[0] != "{" else ("\\sqrt" + split)
  return new_string


def strip_string(string: str) -> str:
  """The reference MATH normalisation (`math_equivalence._strip_string`)."""
  string = string.replace("\n", "").replace("\\!", "").replace("\\\\", "\\")
  string = string.replace("tfrac", "frac").replace("dfrac", "frac")
  string = string.replace("\\left", "").replace("\\right", "")
  string = string.replace("^{\\circ}", "").replace("^\\circ", "")
  string = string.replace("\\$", "")
  string = _remove_right_units(string)
  string = string.replace("\\%", "")
  string = string.replace(" .", " 0.").replace("{.", "{0.")
  if not string:
    return string
  if string[0] == ".":
    string = "0" + string
  if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
    string = string.split("=")[1]
  string = _fix_sqrt(string)
  string = string.replace(" ", "")
  string = _fix_fracs(string)
  if string == "0.5":
    string = "\\frac{1}{2}"
  return _fix_a_slash_b(string)


def normalize_answer(answer: str | None) -> str:
  """The normalised form under which two answers are the same; the raw answer when the reference
  normalisation cannot apply (as in the reference `is_equiv`)."""
  if answer is None:
    return ""
  try:
    return strip_string(answer)
  except _Unnormalisable:
    return answer


def score(pred: str | None, gold: str) -> tuple[bool, str]:
  """(correct, error) for one extracted answer. Error "no_answer" when nothing was extracted."""
  if pred is None or not pred.strip():
    return False, "no_answer"
  return normalize_answer(pred) == normalize_answer(gold), ""


def gold_response(item: Item) -> str:
  return f"The answer follows from the problem.\n\\boxed{{{item.gold}}}"


# ---------------------------------------------------------------------------
# Pipeline entry points (prompts.py, grade.py)
# ---------------------------------------------------------------------------


def prompt_rows(root: Path, split: str, items: list[Item]) -> tuple[list[dict], dict]:
  """One prompt row per item and the task-specific manifest fields (problem fingerprints, subjects)."""
  del root
  rows = [{"id": it.id, "problem_hash": it.problem_hash, "split": split, "subject": it.subject, "level": it.level,
           "source": it.source, "messages": messages(it)} for it in items]
  subjects: dict[str, int] = {}
  for it in items:
    subjects[it.subject] = subjects.get(it.subject, 0) + 1
  extra = {"problems": sorted({it.problem_hash for it in items}), "subjects": dict(sorted(subjects.items())),
           "prompt_version": PROMPT_VERSION}
  return rows, extra


def control_samples(items: dict[str, Item], kind: str, seed: int = 0) -> dict[str, list[tuple[int, str | None]]]:
  """Evaluator controls, each passed through `extract_answer` as a model response would be.

  gold: each problem's gold answer in a response (`gold_response`), graded against itself; must
  score 100%. shuffled: each problem gets the gold answer of another problem of the same subject with
  a different normalised gold answer, drawn with random.Random(seed); a negative control that must
  score low.
  """
  if kind == "gold":
    return {i: [(0, extract_answer(gold_response(it)))] for i, it in items.items()}
  if kind != "shuffled":
    raise ValueError(f"unknown control {kind!r}")
  rng = random.Random(seed)
  by_subject: dict[str, list[Item]] = {}
  for it in items.values():
    by_subject.setdefault(it.subject, []).append(it)
  norm = {it.id: normalize_answer(it.gold) for it in items.values()}
  out: dict[str, list[tuple[int, str | None]]] = {}
  for it in items.values():
    rest = [o for o in by_subject[it.subject] if norm[o.id] != norm[it.id]]
    if rest:
      out[it.id] = [(0, extract_answer(gold_response(rng.choice(rest))))]
  return out


def grade(root: Path, split: str, items: dict[str, Item], samples: dict[str, list[tuple[int, str | None]]], *,
          workers: int = 0, timeout: float = 0.0) -> list[dict]:
  """Graded rows {id, problem_hash, subject, level, sample, pred_answer, correct, error}."""
  del root, split, workers, timeout
  rows = []
  for qid, preds in samples.items():
    it = items[qid]
    for idx, pred in preds:
      ok, err = score(pred, it.gold)
      rows.append({"id": qid, "problem_hash": it.problem_hash, "subject": it.subject, "level": it.level,
                   "sample": idx, "pred_answer": pred, "correct": ok, "error": err})
  return rows
