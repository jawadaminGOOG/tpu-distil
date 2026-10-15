#!/usr/bin/env python3
"""Render the project poster (SVG, optionally PNG) from a results JSON.

The numbers come from `results.json` beside this script, which copies values out of the generated
reports (`distil/report.py`, `distil/compute_report.py`) and trainer logs; the script draws, it
does not compute results. Fonts are embedded as data URIs when `--fonts` points at a directory
holding Caprasimo-Regular.ttf and Figtree[wght].ttf.

Usage: make_poster.py --results results.json --out ../tpu_distil_poster.svg [--fonts DIR] [--png out.png]
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from xml.sax.saxutils import escape

W, H = 2400, 1500
# Every hue is a colour from Sanzo Wada's "A Dictionary of Color Combinations", chosen nearest to the
# Google brand colours, set on a warm paper background.
PAPER, SURFACE, RULE = "#f6f1e6", "#ece4d3", "#ddd3bf"
INK, MUTED, GRID = "#111314", "#635a3a", "#d3c8b2"  # Black; Deep Grayish Olive
ACCENT = "#006eb8"  # Blue
# Brand accents for decoration only (badges, tile stripes): Blue, Spectrum Red, Orange Yellow, Green.
# Orange Yellow, Spectrum Red and Blue form Wada combination 257.
BRAND = ["#006eb8", "#e31f26", "#fcb315", "#489b6e"]
# Data colours per arm, categorical (validated on SURFACE, all pairs): Sulphine Yellow, Spectrum Red,
# Blue, Green. Sulphine Yellow replaces Orange Yellow because Orange Yellow is too light for a mark.
ARM = {"base": "#c19f2c", "self": "#e31f26", "kd": "#006eb8", "rl": "#489b6e"}
# Text drawn on top of each fill.
ON = {"#c19f2c": INK, "#e31f26": PAPER, "#006eb8": PAPER, "#489b6e": INK, "#fcb315": INK}
# Ordinal ramp, one hue, light to dark (validated on SURFACE): Olympic Blue, Blue, Deep Lyons Blue,
# Deep Indigo.
FUNNEL = ["#5a82b3", "#006eb8", "#1c4286", "#051230"]
HEAD, BODY = "Caprasimo, Georgia, serif", "Figtree, Helvetica, Arial, sans-serif"


def t(x, y, s, size=17, weight=400, fill=INK, anchor="start", font=BODY, extra=""):
  return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="{font}" font-size="{size}" font-weight="{weight}" '
          f'fill="{fill}" text-anchor="{anchor}" {extra}>{escape(str(s))}</text>')


def card(x, y, w, h, title, sub=None):
  out = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="18" fill="{SURFACE}" stroke="{RULE}" stroke-width="2"/>',
         t(x + 26, y + 44, title, size=27, font=HEAD)]
  if sub:
    out.append(t(x + 26, y + 70, sub, size=15, fill=MUTED))
  return out


def hbar(x0, y, length, h, color):
  r = min(4, max(length, 0.01) / 2, h / 2)
  x1 = x0 + max(length, 0.01)
  return (f'<path d="M{x0:.1f},{y:.1f} H{x1 - r:.1f} Q{x1:.1f},{y:.1f} {x1:.1f},{y + r:.1f} V{y + h - r:.1f} '
          f'Q{x1:.1f},{y + h:.1f} {x1 - r:.1f},{y + h:.1f} H{x0:.1f} Z" fill="{color}"/>')


def vbar(x, y0, w, length, color):
  """Vertical bar standing on baseline y0 with a 4px rounded top."""
  r = min(4, w / 2, max(length, 0.01) / 2)
  y1 = y0 - max(length, 0.01)
  return (f'<path d="M{x:.1f},{y0:.1f} V{y1 + r:.1f} Q{x:.1f},{y1:.1f} {x + r:.1f},{y1:.1f} H{x + w - r:.1f} '
          f'Q{x + w:.1f},{y1:.1f} {x + w:.1f},{y1 + r:.1f} V{y0:.1f} Z" fill="{color}"/>')


def line(x1, y1, x2, y2, color=GRID, wdt=1.5, dash=None):
  d = f' stroke-dasharray="{dash}"' if dash else ""
  return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{wdt}"{d}/>'


def polyline(pts, color, wdt=2.5):
  return (f'<polyline points="{" ".join(f"{a:.1f},{b:.1f}" for a, b in pts)}" fill="none" stroke="{color}" '
          f'stroke-width="{wdt}" stroke-linejoin="round" stroke-linecap="round"/>')


def dot(x, y, color, r=6):
  return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{color}" stroke="{SURFACE}" stroke-width="2"/>'


# ---------------------------------------------------------------- header and pipeline

def header(r):
  o = [t(56, 86, r["title"], size=48, font=HEAD), t(56, 128, r["subtitle"], size=21, fill=INK),
       t(56, 162, r["setup"], size=16, fill=MUTED)]
  n, tw = len(r["kpis"]), 232
  x0 = W - 56 - n * tw - (n - 1) * 16
  for i, (big, l1, l2) in enumerate(r["kpis"]):
    x = x0 + i * (tw + 16)
    o += [f'<rect x="{x}" y="40" width="{tw}" height="140" rx="18" fill="{PAPER}" stroke="{RULE}" stroke-width="2"/>',
          f'<rect x="{x + 20}" y="40" width="{tw - 40}" height="7" rx="3.5" fill="{BRAND[i % 4]}"/>',
          t(x + tw / 2, 100, big, size=42 if len(big) <= 9 else 32 if len(big) <= 12 else 25, font=HEAD, anchor="middle"),
          t(x + tw / 2, 132, l1, size=15.5, weight=700, anchor="middle"),
          t(x + tw / 2, 156, l2, size=13.5, fill=MUTED, anchor="middle")]
  o.append(line(56, 206, W - 56, 206, RULE, 2))
  return o


def pipeline(x, y, w, h, stages, routes):
  """Five stages; each box shows both routes (one block per recipe), coloured by the recipe's arm."""
  o = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="18" fill="{SURFACE}" stroke="{RULE}" stroke-width="2"/>',
       t(x + 26, y + 44, "The two recipes, end to end", size=27, font=HEAD)]
  lx = x + 470
  for ri, rt in enumerate(routes):
    o.append(f'<rect x="{lx + ri * 330}" y="{y + 30}" width="16" height="16" rx="3" fill="{ARM[rt["arm"]]}"/>')
    o.append(t(lx + 24 + ri * 330, y + 43, rt["label"], size=15, weight=700))
  n = len(stages)
  gap = 54
  bw = (w - 52 - gap * (n - 1)) / n
  by, bh = y + 62, h - 82
  for i, s_ in enumerate(stages):
    bx = x + 26 + i * (bw + gap)
    o.append(f'<rect x="{bx:.1f}" y="{by}" width="{bw:.1f}" height="{bh}" rx="14" fill="{PAPER}" stroke="{RULE}" stroke-width="2"/>')
    badge = BRAND[i % 4]
    o.append(f'<circle cx="{bx + 28:.1f}" cy="{by + 28}" r="15" fill="{badge}"/>')
    o.append(t(bx + 28, by + 34, str(i + 1), size=15, weight=800, fill=ON[badge], anchor="middle"))
    o.append(t(bx + 52, by + 34, s_["name"], size=18, weight=800))
    for ri, rt in enumerate(routes):
      blk = s_[rt["key"]]
      yy = by + 56 + ri * 58
      o.append(f'<rect x="{bx + 16:.1f}" y="{yy:.1f}" width="5" height="46" rx="2.5" fill="{ARM[rt["arm"]]}"/>')
      o.append(t(bx + 30, yy + 21, blk["big"], size=22, font=HEAD))
      o.append(t(bx + 30, yy + 41, blk["line"], size=13.5, fill=MUTED))
    if i < n - 1:
      ax = bx + bw + 6
      ay = by + bh / 2
      o.append(f'<path d="M{ax:.1f},{ay:.1f} H{ax + gap - 20:.1f}" stroke="{MUTED}" stroke-width="3"/>')
      o.append(f'<path d="M{ax + gap - 20:.1f},{ay - 7:.1f} L{ax + gap - 7:.1f},{ay:.1f} L{ax + gap - 20:.1f},{ay + 7:.1f} Z" fill="{MUTED}"/>')
      if s_.get("arrow"):
        o.append(t(ax + (gap - 14) / 2, ay - 12, s_["arrow"], size=12, fill=MUTED, anchor="middle"))
  return o


# ---------------------------------------------------------------- charts

def ladder(x, y, w, h, tasks):
  o = card(x, y, w, h, "Accuracy ladder", "Final same-run evaluation per task · dashed line = teacher")
  px0, py0, pw, ph = x + 70, y + h - 70, w - 100, h - 170
  top = 100
  o += [line(px0, py0, px0 + pw, py0, MUTED, 1.5)]
  for v in (20, 40, 60, 80, 100):
    yy = py0 - ph * v / top
    o += [line(px0, yy, px0 + pw, yy, GRID, 1), t(px0 - 10, yy + 5, f"{v}%", size=13, fill=MUTED, anchor="end")]
  gw = pw / len(tasks)
  for gi, task in enumerate(tasks):
    arms = task["arms"]
    gap = 8
    bw = min(64, (gw - 30 - gap * (len(arms) - 1)) / len(arms))
    span = len(arms) * bw + (len(arms) - 1) * gap
    gx = px0 + gi * gw + (gw - span) / 2
    for ai, (k, label, v) in enumerate(arms):
      bx = gx + ai * (bw + gap)
      o.append(vbar(bx, py0, bw, ph * v / top, ARM[k]))
      if ph * (task["teacher"] - v) / top < 32:  # label inside the bar when it would touch the teacher line
        o.append(t(bx + bw / 2, py0 - ph * v / top + 22, f"{v:.1f}", size=15, weight=800, fill=ON[ARM[k]], anchor="middle"))
      else:
        o.append(t(bx + bw / 2, py0 - ph * v / top - 9, f"{v:.1f}", size=15, weight=800, anchor="middle"))
      for li, part in enumerate(label.split("|")):
        o.append(t(bx + bw / 2, py0 + 19 + li * 15, part, size=12.5, fill=MUTED, anchor="middle"))
    ty = py0 - ph * task["teacher"] / top
    o.append(line(gx - 10, ty, gx + span + 10, ty, INK, 2, "7 5"))
    o.append(t(gx + span + 10, ty - 8, f'Teacher {task["teacher"]:.1f}', size=13, anchor="end"))
    o.append(t(gx + span / 2, py0 + 56, task["label"], size=15.5, weight=800, anchor="middle"))
  return o


def decomposition(x, y, w, h, gains, paired):
  o = card(x, y, w, h, "Where the gain comes from", "Points over base by stage, final same-run evaluation per task")
  bx, bw = x + 110, w - 230
  m = max(g["sft"] + g["rl"] for g in gains)
  yy = y + 92
  for g in gains:
    s_len, r_len = max(bw * g["sft"] / m, 3), max(bw * g["rl"] / m, 3)
    o.append(t(x + 26, yy + 24, g["short"], size=17, weight=800))
    o.append(f'<rect x="{bx:.1f}" y="{yy}" width="{s_len - 1:.1f}" height="34" fill="{ARM[g["sft_arm"]]}"/>')
    o.append(hbar(bx + s_len + 1, yy, r_len - 1, 34, ARM["rl"]))
    if s_len > 70:
      o.append(t(bx + s_len / 2, yy + 23, f'+{g["sft"]:.1f}', size=15, weight=800, fill=ON[ARM[g["sft_arm"]]], anchor="middle"))
      o.append(t(bx + s_len + r_len / 2, yy + 23, f'+{g["rl"]:.1f}', size=15, weight=800, fill=ON[ARM["rl"]], anchor="middle"))
      o.append(t(bx + s_len + r_len + 8, yy + 23, f'= +{g["sft"] + g["rl"]:.1f}', size=15, weight=700))
    else:
      o.append(t(bx + s_len + r_len + 8, yy + 23, f'+{g["sft"]:.1f} + {g["rl"]:.1f} = +{g["sft"] + g["rl"]:.1f}', size=15, weight=700))
    yy += 46
  for i, (arm, lab) in enumerate((("kd", "SFT on teacher answers"), ("self", "SFT on own answers"), ("rl", "GRPO"))):
    lx = x + 26 + i * 220
    o.append(f'<rect x="{lx}" y="{yy + 2}" width="15" height="15" rx="3" fill="{ARM[arm]}"/>')
    o.append(t(lx + 22, yy + 14, lab, size=13.5, fill=MUTED))
  yy += 44
  o.append(t(x + 26, yy, "Controls and paired differences, 95% CI", size=16, weight=800))
  for task, label, val, ci in paired:
    yy += 24
    o.append(line(x + 26, yy - 17, x + w - 26, yy - 17, RULE, 1))
    o.append(t(x + 26, yy, task, size=14, weight=700))
    o.append(t(x + 100, yy, label, size=14, fill=MUTED))
    o.append(t(x + w - 170, yy, f"{val:+.1f}", size=14.5, weight=800, anchor="end"))
    o.append(t(x + w - 26, yy, f"[{ci[0]:+.1f}, {ci[1]:+.1f}]", size=14, fill=MUTED, anchor="end"))
  return o


def cost_bars(x, y, w, h, rows):
  o = card(x, y, w, h, "Better and cheaper", "Serving cost per 1,000 correct answers · teacher (8× H200) vs final student (1 v6e chip)")
  lx, bx = x + 26, x + 150
  bw = w - 150 - 210
  m = max(r["teacher"]["cost"] for r in rows)
  yy = y + 98
  for r in rows:
    o.append(t(lx, yy + 30, r["short"], size=17, weight=800))
    o.append(t(lx, yy + 50, r["student"], size=12.5, fill=MUTED))
    for j, (k, col) in enumerate((("teacher", INK), ("final", ARM[r["arm"]]))):
      e = r[k]
      ln = max(bw * e["cost"] / m, 6)
      by = yy + 4 + j * 30
      o.append(hbar(bx, by, ln, 24, col))
      o.append(t(bx + ln + 8, by + 18, f'${e["cost"]:.3f} · {e["acc"]:.1f}%', size=14, weight=700))
    o.append(t(x + w - 26, yy + 42, f'{r["ratio"]:.1f}×', size=30, font=HEAD, fill=ACCENT, anchor="end"))
    yy += 82
  o.append(f'<rect x="{lx}" y="{yy + 6}" width="15" height="15" rx="3" fill="{INK}"/>')
  o.append(t(lx + 22, yy + 18, "teacher, DeepSeek-V4.1-Flash", size=13.5, fill=MUTED))
  o.append(t(lx + 260, yy + 18, "coloured: final student · large number: teacher ÷ student", size=13.5, fill=MUTED))
  return o


def recipes(x, y, w, h, items):
  """Two stacked recipe boxes: which recipe for which student, with the evidence for each."""
  o = card(x, y, w, h, "Two recipes", "The self-distillation control decides which one a student needs")
  bh = (h - 106 - 14) / len(items)
  yy = y + 90
  for it in items:
    col = ARM[it["arm"]]
    o.append(f'<rect x="{x + 26}" y="{yy:.1f}" width="{w - 52}" height="{bh:.1f}" rx="14" fill="{PAPER}" stroke="{RULE}" stroke-width="2"/>')
    o.append(f'<rect x="{x + 26}" y="{yy + 14:.1f}" width="7" height="{bh - 28:.1f}" rx="3.5" fill="{col}"/>')
    o.append(t(x + 48, yy + 34, it["when"], size=14, weight=700, fill=MUTED))
    o.append(t(x + 48, yy + 62, it["recipe"], size=21, font=HEAD))
    for j, ln in enumerate(it["lines"]):
      o.append(t(x + 48, yy + 90 + j * 22, ln, size=14.5, fill=INK if j == 0 else MUTED, weight=700 if j == 0 else 400))
    yy += bh + 14
  return o


def compute(x, y, w, h, rows):
  o = card(x, y, w, h, "What it took", "TPU v6e chip-hours to produce each final student, by stage")
  stages = [("data", "student samples", ARM["self"]), ("sft", "SFT", ARM["kd"]), ("grpo", "GRPO updates", ARM["rl"]),
            ("rollouts", "GRPO rollouts", ARM["base"])]
  bx, bw = x + 110, w - 136
  m = max(sum(r["stages"].get(k, 0) for k, _, _ in stages) for r in rows)
  yy = y + 96
  for r in rows:
    o.append(t(x + 26, yy + 24, r["short"], size=17, weight=800))
    cx = bx
    for k, _, col in stages:
      v = r["stages"].get(k, 0)
      if not v:
        continue
      ln = bw * v / m
      o.append(f'<rect x="{cx:.1f}" y="{yy}" width="{max(ln - 2, 1):.1f}" height="34" rx="3" fill="{col}"/>')
      if ln > 44:
        o.append(t(cx + ln / 2, yy + 23, f"{v:.1f}", size=14, weight=800, fill=ON[col], anchor="middle"))
      cx += ln
    o.append(t(bx, yy + 54, r["note"], size=13.5, fill=MUTED))
    yy += 82
  for i, (k, lab, col) in enumerate(stages):
    lx = x + 26 + i * 180
    o.append(f'<rect x="{lx}" y="{yy + 2}" width="15" height="15" rx="3" fill="{col}"/>')
    o.append(t(lx + 22, yy + 14, lab, size=13.5, fill=MUTED))
  return o


def recipe_card(x, y, w, h, card_):
  o = card(x, y, w, h, "Recipe card")
  c1, c2 = x + w - 26 - 230, x + w - 26
  for (lab, arm), cx in zip(card_["columns"], (c1, c2)):
    o.append(t(cx, y + 44, lab, size=15, weight=800, fill=ARM[arm], anchor="end"))
  yy = y + 84
  for row in card_["rows"]:
    o.append(line(x + 26, yy + 10, x + w - 26, yy + 10, GRID, 1))
    o.append(t(x + 26, yy, row[0], size=14, fill=MUTED))
    if len(row) == 2:
      o.append(t(c2, yy, row[1], size=14, weight=700, anchor="end"))
    else:
      o.append(t(c1, yy, row[1], size=14, weight=700, anchor="end"))
      o.append(t(c2, yy, row[2], size=14, weight=700, anchor="end"))
    yy += 31
  return o


def trust_bar(x, y, w, h, items):
  """One row of short badges: the measurements that make the numbers credible."""
  o = []
  gap = 14
  bw = (w - gap * (len(items) - 1)) / len(items)
  for i, (big, cap) in enumerate(items):
    bx = x + i * (bw + gap)
    o.append(f'<rect x="{bx:.1f}" y="{y}" width="{bw:.1f}" height="{h}" rx="{h / 2:.0f}" fill="{PAPER}" stroke="{RULE}" stroke-width="2"/>')
    o.append(t(bx + 24, y + h / 2 + 9, big, size=24, font=HEAD, fill=ACCENT))
    o.append(t(bx + 24 + 12 * len(big) + 22, y + h / 2 + 6, cap, size=15, weight=700))
  return o


def font_faces(fonts: Path | None) -> str:
  if not fonts:
    return ""
  faces = []
  for fam, fn in (("Caprasimo", "Caprasimo-Regular.ttf"), ("Figtree", "Figtree[wght].ttf")):
    data = base64.b64encode((fonts / fn).read_bytes()).decode()
    rng = "font-weight: 300 900;" if fam == "Figtree" else ""
    faces.append(f"@font-face {{ font-family: '{fam}'; src: url(data:font/ttf;base64,{data}) format('truetype'); {rng} }}")
  return "<style>" + "\n".join(faces) + "</style>"


def render(r: dict, fonts: Path | None) -> str:
  o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">', "<defs>",
       font_faces(fonts), "</defs>", f'<rect width="{W}" height="{H}" fill="{PAPER}"/>']
  o += header(r)
  o += trust_bar(56, 222, W - 112, 54, r["trust"])
  o += pipeline(56, 292, W - 112, 264, r["pipeline"], r["routes"])
  o += ladder(56, 572, 760, 440, r["tasks"])
  o += cost_bars(836, 572, 760, 440, r["cost"])
  o += decomposition(1616, 572, 728, 440, r["gains"], r["paired"])
  o += recipes(56, 1028, 760, 416, r["recipes"])
  o += compute(836, 1028, 760, 416, r["compute"])
  o += recipe_card(1616, 1028, 728, 416, r["recipe_card"])
  o.append("</svg>")
  return "\n".join(o)


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--results", type=Path, required=True)
  ap.add_argument("--out", type=Path, required=True)
  ap.add_argument("--fonts", type=Path, default=None)
  ap.add_argument("--png", type=Path, default=None)
  args = ap.parse_args()
  svg = render(json.loads(args.results.read_text()), args.fonts)
  args.out.write_text(svg)
  if args.png:
    import cairosvg  # pylint: disable=import-outside-toplevel
    cairosvg.svg2png(bytestring=svg.encode(), write_to=str(args.png), output_width=W, output_height=H)


if __name__ == "__main__":
  main()
