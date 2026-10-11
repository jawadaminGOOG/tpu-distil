"""Text overlap between questions: case/space-normalised exact match and shared word n-grams.

Used by prompts.py (to leave overlapping training questions out) and by leak_audit.py (to check that
none remain), so both apply the same normalisation and the same n.
"""
from __future__ import annotations

import re
from typing import Iterable

N = 13


def normalize_text(text: str) -> str:
  """Lower-case words of letters and digits, joined by single spaces."""
  return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def ngrams(text: str, n: int = N) -> set[tuple[str, ...]]:
  w = normalize_text(text).split()
  return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


class OverlapIndex:
  """A reference set (for example the evaluation questions): their normalised text for exact matches,
  and the word n-grams of the questions and of any `extra_texts` (such as BIRD evidence)."""

  def __init__(self, questions: Iterable[str], extra_texts: Iterable[str] = (), n: int = N):
    self.n = n
    self.exact: set[str] = set()
    self.grams: set[tuple[str, ...]] = set()
    for t in questions:
      self.exact.add(normalize_text(t))
      self.grams |= ngrams(t, n)
    for t in extra_texts:
      self.grams |= ngrams(t, n)

  def matches_exactly(self, text: str) -> bool:
    return normalize_text(text) in self.exact

  def shares_ngram(self, *texts: str) -> bool:
    return any(ngrams(t, self.n) & self.grams for t in texts)
