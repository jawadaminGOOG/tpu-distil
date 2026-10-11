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
          t(x + tw / 2, 100, big, size=42 if len(big) <= 9 else 34, font=HEAD, anchor="middle"),
          t(x + tw / 2, 132, l1, size=15.5, weight=700, anchor="middle"),
          t(x + tw / 2, 156, l2, size=13.5, fill=MUTED, anchor="middle")]
  o.append(line(56, 206, W - 56, 206, RULE, 2))
  return o


def pipeline(x, y, w, h, stages):
  o = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="18" fill="{SURFACE}" stroke="{RULE}" stroke-width="2"/>',
       t(x + 26, y + 44, "The recipe, end to end", size=27, font=HEAD)]
  n = len(stages)
  gap = 70
  bw = (w - 52 - gap * (n - 1)) / n
  by, bh = y + 66, h - 90
  for i, s in enumerate(stages):
    bx = x + 26 + i * (bw + gap)
    o.append(f'<rect x="{bx:.1f}" y="{by}" width="{bw:.1f}" height="{bh}" rx="14" fill="{PAPER}" stroke="{RULE}" stroke-width="2"/>')
    badge = BRAND[i % 4]
    o.append(f'<circle cx="{bx + 30:.1f}" cy="{by + 32}" r="17" fill="{badge}"/>')
    o.append(t(bx + 30, by + 38, str(i + 1), size=17, weight=800, fill=ON[badge], anchor="middle"))
    o.append(t(bx + 58, by + 39, s["name"], size=20, weight=800))
    o.append(t(bx + 18, by + 80, s["big"], size=30, font=HEAD))
    for j, ln in enumerate(s["lines"]):
      o.append(t(bx + 18, by + 110 + j * 21, ln, size=14.5, fill=MUTED))
    if i < n - 1:
      ax = bx + bw + 8
      ay = by + bh / 2
      o.append(f'<path d="M{ax:.1f},{ay:.1f} H{ax + gap - 22:.1f}" stroke="{MUTED}" stroke-width="3"/>')
      o.append(f'<path d="M{ax + gap - 22:.1f},{ay - 8:.1f} L{ax + gap - 8:.1f},{ay:.1f} L{ax + gap - 22:.1f},{ay + 8:.1f} Z" fill="{MUTED}"/>')
      if s.get("arrow"):
        o.append(t(ax + (gap - 16) / 2, ay - 14, s["arrow"], size=12.5, fill=MUTED, anchor="middle"))
  return o


# ---------------------------------------------------------------- charts

def ladder(x, y, w, h, tasks):
  o = card(x, y, w, h, "Accuracy ladder", "Final same-run evaluation per task · dashed line = teacher")
  px0, py0, pw, ph = x + 70, y + h - 70, w - 100, h - 170
  o += [line(px0, py0, px0 + pw, py0, MUTED, 1.5)]
  for v in (20, 40, 60, 80):
    yy = py0 - ph * v / 80
    o += [line(px0, yy, px0 + pw, yy, GRID, 1), t(px0 - 10, yy + 5, f"{v}%", size=13, fill=MUTED, anchor="end")]
  gw = pw / len(tasks)
  for gi, task in enumerate(tasks):
    arms = [("base", task["ladder"]["base"]), ("kd", task["ladder"]["kd"]), ("rl", task["ladder"].get("rl"))]
    bw = 70
    gx = px0 + gi * gw + (gw - 3 * bw - 2 * 14) / 2
    for ai, (k, v) in enumerate(arms):
      bx = gx + ai * (bw + 14)
      if v is None:
        o.append(f'<rect x="{bx:.1f}" y="{py0 - ph * 0.5:.1f}" width="{bw}" height="{ph * 0.5:.1f}" rx="6" fill="none" stroke="{MUTED}" stroke-width="2" stroke-dasharray="6 5"/>')
        o.append(t(bx + bw / 2, py0 - ph * 0.5 - 10, "running", size=13, fill=MUTED, anchor="middle"))
      else:
        o.append(vbar(bx, py0, bw, ph * v / 80, ARM[k]))
        if ph * (task["teacher"] - v) / 80 < 20:  # label inside the bar when it would touch the teacher line
          o.append(t(bx + bw / 2, py0 - ph * v / 80 + 24, f"{v:.1f}", size=17, weight=800, fill=ON[ARM[k]], anchor="middle"))
        else:
          o.append(t(bx + bw / 2, py0 - ph * v / 80 - 10, f"{v:.1f}", size=17, weight=800, anchor="middle"))
      o.append(t(bx + bw / 2, py0 + 22, {"base": "Base", "kd": "KD", "rl": "KD+GRPO"}[k], size=14, fill=MUTED, anchor="middle"))
    ty = py0 - ph * task["teacher"] / 80
    o.append(line(gx - 12, ty, gx + 3 * bw + 2 * 14 + 12, ty, INK, 2, "7 5"))
    o.append(t(gx + 3 * bw + 2 * 14 + 12, ty - 8, f'Teacher {task["teacher"]:.1f}', size=13.5, anchor="end"))
    o.append(t(gx + (3 * bw + 28) / 2, py0 + 48, task["label"], size=17, weight=800, anchor="middle"))
  return o


def decomposition(x, y, w, h, tasks, paired):
  o = card(x, y, w, h, "Where the gain comes from", "KD gain over base, split by the self-distillation control")
  bx, bw = x + 110, w - 190
  m = max(tk["decomp"]["total"] for tk in tasks)
  yy = y + 96
  for tk in tasks:
    d = tk["decomp"]
    s_len, t_len = bw * d["self"] / m, bw * d["teacher"] / m
    o.append(t(x + 26, yy + 25, tk["short"], size=17, weight=800))
    o.append(f'<rect x="{bx:.1f}" y="{yy}" width="{s_len - 1:.1f}" height="36" fill="{ARM["self"]}"/>')
    o.append(hbar(bx + s_len + 1, yy, t_len - 1, 36, ARM["kd"]))
    o.append(t(bx + s_len / 2, yy + 24, f'+{d["self"]:.1f}', size=15, weight=800, fill=ON[ARM["self"]], anchor="middle"))
    o.append(t(bx + s_len + t_len / 2, yy + 24, f'+{d["teacher"]:.1f}', size=15, weight=800, fill=ON[ARM["kd"]], anchor="middle"))
    o.append(t(bx + s_len + t_len + 8, yy + 24, f'= +{d["total"]:.1f}', size=15, weight=700))
    yy += 52
  o.append(f'<rect x="{x + 26}" y="{yy + 4}" width="16" height="16" rx="3" fill="{ARM["self"]}"/>')
  o.append(t(x + 50, yy + 17, "in-domain fine-tuning (self-distillation)", size=14, fill=MUTED))
  o.append(f'<rect x="{x + 360}" y="{yy + 4}" width="16" height="16" rx="3" fill="{ARM["kd"]}"/>')
  o.append(t(x + 384, yy + 17, "added by the teacher's answers", size=14, fill=MUTED))
  yy += 62
  o.append(t(x + 26, yy, "Paired differences, 95% CI (KD rows: control runs)", size=16, weight=800))
  for task, label, val, ci in paired:
    yy += 26
    o.append(line(x + 26, yy - 18, x + w - 26, yy - 18, RULE, 1))
    o.append(t(x + 26, yy, task, size=14.5, weight=700))
    o.append(t(x + 110, yy, label, size=14.5, fill=MUTED))
    o.append(t(x + w - 180, yy, f"{val:+.1f}", size=15, weight=800, anchor="end"))
    o.append(t(x + w - 26, yy, f"[{ci[0]:+.1f}, {ci[1]:+.1f}]", size=14.5, fill=MUTED, anchor="end"))
  return o


def cost_bars(x, y, w, h, pts, ratio):
  o = card(x, y, w, h, "Better and cheaper", "BIRD serving cost per 1,000 correct answers · accuracy under each name")
  lx, bx = x + 26, x + 270
  bw = w - 270 - 110
  m = max(p["cost"] for p in pts)
  yy = y + 100
  for p in pts:
    col = INK if p["k"] == "teacher" else ARM[p["k"]]
    o.append(t(lx, yy + 17, p["label"], size=16, weight=700))
    o.append(t(lx, yy + 37, f'{p["hw"]} · {p["acc"]:.1f}%', size=13.5, fill=MUTED))
    ln = max(bw * p["cost"] / m, 6)
    o.append(hbar(bx, yy + 4, ln, 34, col))
    o.append(t(bx + ln + 10, yy + 27, f'${p["cost"]:.3f}', size=16, weight=800))
    yy += 58
  o.append(line(lx, yy + 6, x + w - 26, yy + 6, RULE, 1.5))
  o.append(t(lx, yy + 62, f"{ratio:.1f}×", size=50, font=HEAD, fill=ACCENT))
  o.append(t(lx + 170, yy + 42, "cheaper per correct answer than the teacher,", size=16, weight=700))
  o.append(t(lx + 170, yy + 64, "and accuracy rises at every training stage", size=16, fill=MUTED))
  return o


def funnel(x, y, w, h, f):
  o = card(x, y, w, h, "Teacher data funnel", "Answers kept only when the task verifies them")
  stages = ["teacher answers", "verified correct", "distinct per question", "used for KD"]
  bx, bw = x + 214, w - 300
  m = max(max(v) for v in f.values())
  yy = y + 100
  for task, vals in f.items():
    o.append(t(x + 26, yy - 6, task, size=17, weight=800))
    for i, (st, v) in enumerate(zip(stages, vals)):
      col = FUNNEL[i]
      o.append(t(x + 26, yy + 19 + i * 30, st, size=13.5, fill=MUTED))
      o.append(hbar(bx, yy + 4 + i * 30, bw * v / m, 24, col))
      o.append(t(bx + bw * v / m + 8, yy + 21 + i * 30, f"{v:,}", size=14, weight=700))
    yy += 150
  return o


def compute(x, y, w, h, c):
  o = card(x, y, w, h, "What it took", "BIRD recipe: accelerator-hours by stage")
  segs = c["stages"]
  total = sum(v for _, v in segs)
  bx, bw, by = x + 26, w - 52, y + 98
  cols = [ARM["kd"], ARM["rl"], ARM["base"]]
  cx = bx
  for i, (name, v) in enumerate(segs):
    ln = bw * v / total
    o.append(f'<rect x="{cx:.1f}" y="{by}" width="{ln - 2:.1f}" height="40" rx="4" fill="{cols[i]}"/>')
    o.append(t(cx + ln / 2, by + 26, f"{v:.1f}", size=15, weight=800, fill=ON[cols[i]], anchor="middle"))
    o.append(f'<rect x="{bx}" y="{by + 60 + i * 26}" width="14" height="14" rx="3" fill="{cols[i]}"/>')
    o.append(t(bx + 22, by + 72 + i * 26, f"{name} (v6e chip-hours)", size=14, fill=MUTED))
    cx += ln
  ky = by + 170
  for i, (big, cap) in enumerate(c["tiles"]):
    tx = x + 26 + (i % 2) * (w - 52) / 2
    yy = ky + (i // 2) * 74
    o.append(t(tx, yy + 32, big, size=32, font=HEAD))
    o.append(t(tx, yy + 56, cap, size=13.5, fill=MUTED))
  return o


def recipe_card(x, y, w, h, rows):
  o = card(x, y, w, h, "Recipe card")
  yy = y + 82
  for k, v in rows:
    o.append(line(x + 26, yy + 10, x + w - 26, yy + 10, GRID, 1))
    o.append(t(x + 26, yy, k, size=14.5, fill=MUTED))
    o.append(t(x + w - 26, yy, v, size=14.5, weight=700, anchor="end"))
    yy += 33
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
  o += pipeline(56, 292, W - 112, 264, r["pipeline"])
  o += ladder(56, 572, 760, 440, r["tasks"])
  o += cost_bars(836, 572, 760, 440, r["cost"], r["cost_ratio"])
  o += decomposition(1616, 572, 728, 440, r["tasks"], r["paired"])
  o += funnel(56, 1028, 760, 416, r["funnel"])
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
