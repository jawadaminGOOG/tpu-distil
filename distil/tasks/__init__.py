"""Task registry: every pipeline stage selects its task with `--task` (default `bird`).

A task is a module that provides:
  NAME, LABEL                 short name and display name
  SPLITS, EVAL_SPLIT          the splits it can load and the one that is never trained, tuned or selected on
  FULL_COVERAGE_SPLITS        splits where grade.py counts a question without a generation as incorrect
  GROUP_KEY, GROUP_SHORT,     the key under which prompt, graded and training rows name the question's
  GROUP_NOUN                  group (BIRD: the database; FinQA: the source report page;
                              MATH: a fingerprint of the problem text), a short form used
                              in manifest keys, and the noun used in messages
  PRED_KEY, ANSWER_SHORT,     the key under which graded rows hold the extracted answer, and the short form
  ANSWER_NOUN                 and noun used in counters and messages
  valid_group(name)           whether a string has the form of a group name (checks forbid lists read from files)
  Item                        dataclass with `id` ("<split>-<n>"), `group`, `question` and `gold`
  load_items(root, split)     the questions of one split
  messages(item, context)     the [system, user] chat for one question; prompt_template_sha256()
  prompt_rows(root, split, items) -> (rows, manifest fields)   used by prompts.py
  extract_answer(text)        the final answer of a response, or None
  normalize_answer(answer)    the form under which two answers to one question count as the same
  control_samples(items, kind)  the evaluator controls "gold" and "shuffled"
  grade(root, split, items, samples, workers=, timeout=)  graded rows, one per (question, sample)
"""
from __future__ import annotations

import importlib
from types import ModuleType

NAMES = ("bird", "finqa", "math")
# Task name -> module name. MATH lives in hendrycks_math.py so that no module shadows the stdlib `math`.
_MODULES = {"bird": "bird", "finqa": "finqa", "math": "hendrycks_math"}


def get_task(name: str) -> ModuleType:
  if name not in NAMES:
    raise ValueError(f"unknown task {name!r}; known tasks: {', '.join(NAMES)}")
  return importlib.import_module(f".{_MODULES[name]}", __name__)
