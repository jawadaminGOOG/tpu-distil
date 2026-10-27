#!/usr/bin/env python3
"""Extract the drawing-task figures used by the talk from the drawing experiment's slide export.

Writes docs/talk/assets/: one PNG per rendered SVG (prompt x model) and the distillation loss curve. The renders are
cropped from the comparison rows of the export; nothing is redrawn. Needs PyMuPDF (`pip install pymupdf`).

Usage: extract_drawing_assets.py [--pdf PATH] [--out DIR]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pymupdf as fitz

HERE = Path(__file__).resolve().parent
PDF = HERE.parent / "Unlocking TPU Performance Through Model Distillation.pdf"

# (page, image index on that page, x and y offset of its tiles) of each comparison row, in prompt order
ROWS = {"tree": (6, 5, 0, 0), "star": (6, 6, 0, 0), "sun": (7, 0, 6, 5), "rocket": (7, 3, 0, 0)}
MODELS = ("teacher", "base", "distilled")       # column order in every row: Qwen2.5-Coder-7B, Gemma 3 4B base, Gemma 3 4B distilled
TILE_X = (484, 1060, 1636)                       # left edge of each render tile in a 2048-px-wide row, inside its frame
TILE_Y, TILE = 24, 264
CURVE = (6, 0)                                   # the QLoRA loss curve, plotted as 1 - loss


def image(doc, page, index):
  xref = doc[page - 1].get_images(full=True)[index][0]
  pix = fitz.Pixmap(doc, xref)
  return fitz.Pixmap(fitz.csRGB, pix) if pix.n - pix.alpha > 3 else pix


def crop(pix, x, y, w, h):
  out = fitz.Pixmap(pix.colorspace, fitz.IRect(x, y, x + w, y + h), pix.alpha)
  out.copy(pix, out.irect)
  return out


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--pdf", type=Path, default=PDF)
  ap.add_argument("--out", type=Path, default=HERE / "assets")
  args = ap.parse_args()
  args.out.mkdir(exist_ok=True)
  doc = fitz.open(args.pdf)
  for prompt, (page, index, dx, dy) in ROWS.items():
    row = image(doc, page, index)
    assert row.width == 2048, (prompt, row.width)
    for model, x in zip(MODELS, TILE_X):
      crop(row, x + dx, TILE_Y + dy, TILE, TILE).save(args.out / f"svg_{prompt}_{model}.png")
  curve = image(doc, *CURVE)
  curve.shrink(1)
  curve.save(args.out / "svg_loss_curve.png")
  print(f"wrote {len(ROWS) * len(MODELS) + 1} files to {args.out}")


if __name__ == "__main__":
  main()
