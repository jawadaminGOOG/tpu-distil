#!/usr/bin/env python3
"""Build the TPU-Distil talk (BIRD, FinQA, MATH and Fred's drawing experiment) as one self-contained HTML deck.

Slides are 1920x1080 and laid out on a 12-column grid (120 px margins, 40 px gutters); every element is
an absolutely positioned box, so nothing depends on text flow. Colours come from Sanzo Wada's "A
Dictionary of Color Combinations": Cotinga Purple dark slides, white content slides, and Diamine
Green, Khaki, Eupatorium Purple and Blue Violet as chart accents. Numbers
are copied from the repository's README, docs/RECIPE.md and docs/COMPUTE.md.

Browser keys: Right / space / PageDown / click reveal the next build group, then the next slide; Left / PageUp step back.
Down / Up jump a whole slide, Home / End, N toggles speaker notes, P prints (print shows every slide fully built).
With ?qa in the URL the page writes every box whose content overflows into #qa (used by the build check).

Usage: build_deck.py --out tpu_distil_talk.html --fonts DIR [--notes-md speaker_notes.md]
DIR holds Figtree[wght].ttf (body) and InterTight[wght].ttf (headings), both from Google Fonts; they are embedded in the page.
"""
from __future__ import annotations

import argparse
import base64
import html
import re
from pathlib import Path

# ------------------------------------------------------------------ palette (Sanzo Wada)
# Aubergine dark slides, white content slides, white panels outlined in Fawn. Every colour is a named Wada swatch; the four
# chart accents pass the dataviz validator on white (normal-vision ΔE ≥ 15, protan/deutan ΔE ≥ 8).
DARK = "#501345"      # Cotinga Purple: dark slides, titles
PAPER = "#ffffff"     # White: content-slide background; panels get a Fawn outline
CREAM = "#f5ecc2"     # Sulpher Yellow: text on dark slides, pale highlight
INK = "#253122"       # Deep Slate Olive: body text, teacher
MUTED = "#70727c"     # Deep Violet / Plumbeous: captions on white
SOURCE = INK          # source lines and subheads on the content background
TEAL = "#1a7444"      # Diamine Green: distilled (KD)
CORAL = "#bf5892"     # Eupatorium Purple: hero (+ GRPO)
GOLD = "#bc892b"      # Khaki: self-distillation
BLUE = "#6450a1"      # Blue Violet
SKY = "#b5b1d8"       # Grayish Lavender A: light accent
GRAY = "#a1a39a"      # Warm Gray: untrained baseline
NILE = CREAM          # pale fill (completed steps)
PINK = "#f9c1ce"      # Hermosa Pink: pale warm fill
CARD = "#ffffff"      # White
RULE = "#d1b0a7"      # Fawn: rules, gridlines, inactive progress bars
ARM = {"base": GRAY, "kd": TEAL, "rl": CORAL, "self": GOLD, "corr": SKY, "teacher": INK}
ON = {GRAY: INK, TEAL: "#ffffff", CORAL: "#ffffff", GOLD: INK, SKY: INK, INK: "#ffffff", BLUE: "#ffffff", PAPER: INK, PINK: INK}
CH_COLOR = [CORAL, GOLD, TEAL, BLUE, SKY]   # one accent per talk part, as on its divider
STEP_COLOR = [BLUE, GOLD, SKY, TEAL, CORAL, DARK]   # Measure, Generate, Verify, Distil (KD), Reinforce (GRPO), Serve (the TPU chip)
NEUTRAL = RULE        # top edge of a card that stands for no particular model
ON = {**ON, DARK: "#ffffff", RULE: INK}
ON_DARK = {TEAL: "#b5decc", BLUE: SKY, SKY: PINK}   # Turquoise Green, lavender, pink   # lighter stand-ins where an accent sits on the aubergine

# ------------------------------------------------------------------ grid
W, H, ML, CW, GUT = 1920, 1080, 120, 1680, 40
COLW = (CW - 11 * GUT) / 12
TOP = 250           # content top on a normal slide
BOT = 970           # content bottom


def gx(col: int) -> float:
  return ML + col * (COLW + GUT)


def gw(span: int) -> float:
  return span * COLW + (span - 1) * GUT


SLIDES: list[tuple[str, str, str]] = []   # (body, notes, theme)


def esc(s) -> str:
  return html.escape(str(s), quote=False)


def box(col, span, top, height, inner, cls="", style=""):
  return (f'<div class="box {cls}" style="left:{gx(col):.1f}px;top:{top:.1f}px;width:{gw(span):.1f}px;'
          f'height:{height:.1f}px;{style}">{inner}</div>')


# ------------------------------------------------------------------ SVG helpers

def svg(w, h, body, bare=False):
  cls = ' class="bare"' if bare else ""
  return (f'<svg{cls} viewBox="0 0 {w:.0f} {h:.0f}" width="100%" height="100%" preserveAspectRatio="xMidYMid meet" '
          f'font-family="Figtree, Arial, sans-serif">{body}</svg>')


def T(x, y, s, size=24, weight=400, fill=INK, anchor="start", extra=""):
  return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" fill="{fill}" '
          f'text-anchor="{anchor}" {extra}>{esc(s)}</text>')


def R(x, y, w, h, fill, rx=8, stroke=None, sw=2, dash=None):
  st = f' stroke="{stroke}" stroke-width="{sw}"' if stroke else ""
  d = f' stroke-dasharray="{dash}"' if dash else ""
  return f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(w, 0.5):.1f}" height="{max(h, 0.5):.1f}" rx="{rx}" fill="{fill}"{st}{d}/>'


def Lf(x1, y1, x2, y2, color=INK, w=3, dash=None):
  d = f' stroke-dasharray="{dash}"' if dash else ""
  return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{w}"{d}/>'


def C(x, y, r, fill, stroke=None, sw=3):
  st = f' stroke="{stroke}" stroke-width="{sw}"' if stroke else ""
  return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill}"{st}/>'


def arrow(x1, y, x2, color=MUTED, w=4):
  return (Lf(x1, y, x2 - 14, y, color, w) +
          f'<path d="M{x2 - 16:.1f},{y - 9:.1f} L{x2:.1f},{y:.1f} L{x2 - 16:.1f},{y + 9:.1f} Z" fill="{color}"/>')


# ------------------------------------------------------------------ slide shells

def chapter_track(ch: int) -> str:
  out = []
  x = W - ML - 5 * 46 + 6
  for i in range(5):
    col = CH_COLOR[ch] if i == ch else RULE
    out.append(f'<div class="track" style="left:{x + i * 46}px;background:{col}"></div>')
  return "".join(out)


def add(body, notes, theme="light"):
  SLIDES.append((body, notes, theme))


TITLES: list[tuple[int, str]] = []   # (slide number, plain title) of every main-deck content slide


def plain(s: str) -> str:
  return html.unescape(" ".join(re.sub(r"<[^>]+>", " ", s).split()))


def source_line(source) -> str:
  """One source line, or two stacked lines when a slide covers two topics."""
  if not source:
    return ""
  if isinstance(source, str):
    return f'<div class="source" style="left:{ML}px;width:{CW}px">{source}</div>'
  return f'<div class="source two" style="left:{ML}px;width:{CW}px">{"<br>".join(source)}</div>'


def content(ch, kicker, title, inner, notes, source="", also=()):
  """A content slide. `also` names the part headers on it, so appendix links can target either part."""
  col = CH_COLOR[ch]
  src = source_line(source)
  inner = inner.replace('<div class="hh">', f'<div class="hh" style="border-bottom-color:{col}">')
  add(f'''{chapter_track(ch)}
<div class="kicker" style="left:{ML}px;top:70px"><span class="dot" style="background:{col}"></span>{esc(kicker)}</div>
<h2 class="title" style="left:{ML}px;top:104px;width:{CW}px">{title}</h2>
{inner}{src}''', notes)
  TITLES.append((len(SLIDES), plain(title)))
  TITLES.extend((len(SLIDES), plain(t)) for t in also)


def ref(fragment: str) -> tuple[int, str]:
  """The main-deck slide whose title contains `fragment`, so appendix links follow any reordering."""
  for n, t in TITLES:
    if fragment in t:
      return n, t
  raise KeyError(f"no main slide title contains {fragment!r}")


APPENDIX: list[tuple[str, int, int, str, str]] = []   # (code, appendix slide number, main slide number, title, part colour)


def appendix(ch, code_, fragment, title, inner, notes, source=""):
  """A deep-dive slide: kicker carries its code, a pill names the main slide it backs."""
  n, t = ref(fragment)
  col = DARK
  src = f'<div class="source" style="left:{ML}px;width:{CW}px">{source}</div>' if source else ""
  add(f'''<div class="kicker" style="left:{ML}px;top:70px"><span class="appx" style="background:{col};color:#ffffff">{esc(code_)}</span>Appendix</div>
<div class="backref" style="right:{ML}px;top:62px;border-color:{col}">Supports slide {n} · {esc(t)}</div>
<h2 class="title" style="left:{ML}px;top:104px;width:{CW}px">{title}</h2>
{inner}{src}''', f"Backup for slide {n} ({t}). {notes}")
  APPENDIX.append((code_, len(SLIDES), n, plain(title), col))


def table(headers, rows, widths=None, size=24, bold_last=False, accent=TEAL, cls=""):
  """An HTML table card; cells are trusted HTML."""
  cols = "".join(f'<col style="width:{w}%">' for w in widths) if widths else ""
  head = "".join(f"<th>{h}</th>" for h in headers)
  body = []
  for i, r in enumerate(rows):
    cls = ' class="hl"' if bold_last and i == len(rows) - 1 else ""
    body.append(f"<tr{cls}>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>")
  return (f'<div class="tbl {cls}" style="border-top:10px solid {NEUTRAL};font-size:{size}px"><table><colgroup>{cols}</colgroup>'
          f'<thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>')


def forest(rows, w, h, lo, hi, label_w=330):
  """Point estimates with 95% intervals; rows = [(label, sub, est, low, high, color)]."""
  o = []
  x0, x1 = label_w, w - 40
  sx = lambda v: x0 + (x1 - x0) * (v - lo) / (hi - lo)  # noqa: E731
  step = (h - 60) / len(rows)
  for v in range(lo, hi + 1, 5):
    o.append(Lf(sx(v), 0, sx(v), h - 60, RULE if v else INK, 2 if v else 3))
    o.append(T(sx(v), h - 22, f"{v:+d}" if v else "0", 20, fill=MUTED, anchor="middle"))
  for i, (lab, sub, est, l, u, col) in enumerate(rows):
    y = i * step + step / 2
    o.append(T(0, y - 2, lab, 26, 700))
    o.append(T(0, y + 26, sub, 20, fill=MUTED))
    o.append(Lf(sx(l), y, sx(u), y, col, 10))
    o.append(C(sx(est), y, 13, col, PAPER, 4))
    o.append(T(sx(u) + 18, y + 9, f"{est:+.1f}", 26, 800))
  return svg(w, h, "".join(o))


def vbars(rows, w, h, vmax, fmt="{:.0f}"):
  """Vertical bars; rows = [(label, value, color)]."""
  o = []
  y0, ph = h - 50, h - 110
  slot = w / len(rows)
  bw = slot * 0.6
  o.append(Lf(0, y0, w, y0, INK, 3))
  for i, (lab, v, col) in enumerate(rows):
    x = slot * i + (slot - bw) / 2
    bh = ph * v / vmax
    o.append(R(x, y0 - bh, bw, bh, col, 8))
    o.append(T(x + bw / 2, y0 - bh - 14, fmt.format(v), 26, 800, anchor="middle"))
    o.append(T(x + bw / 2, y0 + 34, lab, 22, 600, MUTED, "middle"))
  return svg(w, h, "".join(o))


def dark_divider(num, title, subtitle, accent, notes):
  accent = ON_DARK.get(accent, accent)
  third = CORAL if accent != CORAL else GOLD
  add(f'''<div class="bignum" style="left:{ML - 8}px;top:110px;-webkit-text-stroke:3px {accent}">{num:02d}</div>
<h1 class="div-title" style="left:{ML}px;top:450px;width:1080px">{title}</h1>
<div class="div-sub" style="left:{ML}px;top:760px;width:980px">{esc(subtitle)}</div>
<div class="circle" style="left:1290px;top:170px;width:520px;height:520px;background:{accent}"></div>
<div class="circle" style="left:1210px;top:610px;width:200px;height:200px;background:{NILE};opacity:.92"></div>
<div class="circle" style="left:1640px;top:700px;width:110px;height:110px;background:{third}"></div>''', notes, "dark")


def statement(kicker, text, sub, notes, dark=False, accent=CORAL):
  add(f'''<div class="kicker" style="left:{ML}px;top:150px;color:{accent}"><span class="dot" style="background:{accent}"></span>{esc(kicker)}</div>
<div class="statement" style="left:{ML}px;top:300px;width:1560px">{text}</div>
<div class="div-sub" style="left:{ML}px;top:740px;width:1400px">{sub}</div>''', notes, "dark" if dark else "light")


# ------------------------------------------------------------------ components

def card(title, body, accent=None, tag=None, cls=""):
  accent = accent or NEUTRAL
  t = f'<div class="tag" style="background:{accent};color:{ON.get(accent, INK)}">{esc(tag)}</div>' if tag else ""
  return f'<div class="card {cls}" style="border-top:10px solid {accent}">{t}<div class="card-t">{title}</div><div class="card-b">{body}</div></div>'


def learn(text, accent=None, label="What we learned", cls=""):
  accent = DARK  # lessons are structure, never a model colour
  return (f'<div class="learn {cls}" style="border-left:10px solid {accent}"><div class="learn-l" style="color:{accent}">'
          f'{esc(label)}</div><div class="learn-b">{text}</div></div>')


def stat(big, cap, accent=TEAL, cls=""):
  return f'<div class="stat {cls}"><div class="stat-big" style="color:{accent}">{big}</div><div class="stat-cap">{cap}</div></div>'


def step_track(active: int) -> str:
  names = ["Measure", "Generate", "Verify", "Distil", "Reinforce", "Serve"]
  out = []
  pw = (CW - 5 * 24) / 6
  for i, n in enumerate(names):
    x = ML + i * (pw + 24)
    on, done = i == active, i < active
    bg = DARK if on else (NILE if done else CARD)
    fg = "#ffffff" if on else INK
    bd = DARK if (on or done) else RULE
    out.append(f'<div class="step" style="left:{x:.1f}px;top:{TOP - 20}px;width:{pw:.1f}px;background:{bg};color:{fg};'
               f'border:2px solid {bd}"><b>{i + 1}</b>&nbsp;&nbsp;{n}</div>')
  return "".join(out)


def journey(i, title, visual, did, decision, learned, notes, source=""):
  """One training step: track on top, visual left (7 cols), three stacked panels right (5 cols)."""
  top = TOP + 70
  inner = step_track(i)
  inner += box(0, 7, top, BOT - top, visual, "vis")
  gap = 20
  h3 = (BOT - top - 2 * gap) / 3
  inner += box(7, 5, top, h3, f'<div class="pl">What we did</div><div class="pb">{did}</div>', "panel")
  inner += box(7, 5, top + h3 + gap, h3, f'<div class="pl" style="color:{BLUE}">Decision</div><div class="pb">{decision}</div>', "panel")
  inner += box(7, 5, top + 2 * (h3 + gap), h3, f'<div class="pl" style="color:{DARK}">What we learned</div><div class="pb">{learned}</div>', "panel")
  content(2, f"The journey · step {i + 1} of 6", title, inner, notes, source)


def code(text, accent=CORAL, label="", cls=""):
  light = {CORAL: PINK, TEAL: "#b5decc", BLUE: SKY, GOLD: CREAM, GRAY: CREAM}.get(accent, CREAM)
  lab = f'<div class="code-l" style="color:{light}">{label}</div>' if label else ""
  return f'<div class="code {cls}" style="border-top:8px solid {accent}">{lab}<pre>{esc(text)}</pre></div>'


# ------------------------------------------------------------------ charts

def hbars(rows, w, h, vmax, label_w=330, value_fmt="{:.1f}%", bar_h=56):
  """rows: [(label, sublabel, value, color)]."""
  o = []
  x0, x1 = label_w, w - 170
  step = h / len(rows)
  for i, (lab, sub, v, col) in enumerate(rows):
    y = i * step + (step - bar_h) / 2
    o.append(T(0, y + bar_h / 2 - (4 if sub else -9), lab, 28, 700))
    if sub:
      o.append(T(0, y + bar_h / 2 + 26, sub, 21, fill=MUTED))
    ln = (x1 - x0) * v / vmax
    o.append(R(x0, y, ln, bar_h, col, 10))
    o.append(T(x0 + ln + 14, y + bar_h / 2 + 11, value_fmt.format(v), 30, 800))
  return svg(w, h, "".join(o))


def ladder(groups, w, h, top=100, teacher=None, teacher_left=False, inside=False):
  o = []
  y0, ph = h - 70, h - 130
  x0 = 80
  for v in (25, 50, 75, 100):
    yy = y0 - ph * v / top
    o.append(Lf(x0, yy, w, yy, RULE, 2))
    o.append(T(x0 - 14, yy + 8, f"{v}%", 20, fill=MUTED, anchor="end"))
  o.append(Lf(x0, y0, w, y0, INK, 3))
  slot = (w - x0) / len(groups)
  bw = min(180, slot * (0.78 if inside else 0.58))
  for i, (arm, lab, v) in enumerate(groups):
    bx = x0 + slot * i + (slot - bw) / 2
    bh = ph * v / top
    o.append(f'<path d="M{bx:.1f},{y0:.1f} V{y0 - bh + 12:.1f} Q{bx:.1f},{y0 - bh:.1f} {bx + 12:.1f},{y0 - bh:.1f} '
             f'H{bx + bw - 12:.1f} Q{bx + bw:.1f},{y0 - bh:.1f} {bx + bw:.1f},{y0 - bh + 12:.1f} V{y0:.1f} Z" fill="{ARM[arm]}"/>')
    if inside:
      o.append(T(bx + bw / 2, y0 - bh + 46, f"{v:.1f}%", 30, 800, ON.get(ARM[arm], INK), "middle"))
    else:
      o.append(T(bx + bw / 2, y0 - bh - 20, f"{v:.1f}%", 40, 800, anchor="middle"))
    o.append(T(bx + bw / 2, y0 + 44, lab, 28, 700, anchor="middle"))
  if teacher:
    ty = y0 - ph * teacher / top
    o.append(Lf(x0, ty, w, ty, INK, 3, "12 9"))
    if teacher_left:
      o.append(T(x0 + 8, ty - 16, f"teacher {teacher:.1f}%", 26, 700))
    else:
      o.append(T(w - 4, ty - 16, f"teacher {teacher:.1f}%", 26, 700, anchor="end"))
  return svg(w, h, "".join(o))


def quadrant(points, w, h, xmax=1.6, goal=True):
  """Cost (x, $ per 1,000 correct answers) vs accuracy (y, %)."""
  o = []
  x0, y0, x1, y1 = 120, h - 90, w - 40, 30
  sx = lambda v: x0 + (x1 - x0) * v / xmax  # noqa: E731
  sy = lambda v: y0 - (y0 - y1) * v / 80  # noqa: E731
  if goal:
    o.append(R(sx(0) + 4, sy(80), sx(0.42) - sx(0), sy(44) - sy(80), CREAM, 16))
    o.append(T(sx(0) + 22, sy(76), "the goal:", 22, 700, DARK))
    o.append(T(sx(0) + 22, sy(76) + max(28, sy(71.5) - sy(76)), "cheap and accurate", 22, 700, DARK))
  for v in (20, 40, 60, 80):
    o.append(Lf(x0, sy(v), x1, sy(v), RULE, 1.5))
    o.append(T(x0 - 14, sy(v) + 8, f"{v}%", 20, fill=MUTED, anchor="end"))
  o.append(Lf(x0, y0, x1, y0, INK, 3))
  o.append(Lf(x0, y0, x0, y1, INK, 3))
  for v in (0.4, 0.8, 1.2, 1.6):
    o.append(T(sx(v), y0 + 34, f"${v:.2f}", 20, fill=MUTED, anchor="middle"))
  o.append(T((x0 + x1) / 2, y0 + 72, "serving cost per 1,000 correct answers", 22, 600, MUTED, "middle"))
  o.append(T(36, (y0 + y1) / 2, "accuracy", 22, 600, MUTED, "middle", f'transform="rotate(-90 36 {(y0 + y1) / 2:.1f})"'))
  for lab, cost, acc, col, anchor in points:
    o.append(C(sx(cost), sy(acc), 17, col, PAPER, 4))
    dx = 28 if anchor == "start" else -28
    o.append(T(sx(cost) + dx, sy(acc) + 9, lab, 27, 800, INK, anchor))
  return svg(w, h, "".join(o))


def line_chart(series, xs, w, h, lo, hi, xlabel, yfmt="{:.0f}"):
  o = []
  x0, x1, y0, y1 = 80, w - 200, h - 90, 30
  sx = lambda i: x0 + (x1 - x0) * i / (len(xs) - 1)  # noqa: E731
  sy = lambda v: y0 - (y0 - y1) * (v - lo) / (hi - lo)  # noqa: E731
  for v in range(lo, hi + 1, (hi - lo) // 5):
    o.append(Lf(x0, sy(v), x1, sy(v), RULE, 1.5))
    o.append(T(x0 - 14, sy(v) + 7, yfmt.format(v), 20, fill=MUTED, anchor="end"))
  o.append(Lf(x0, y0, x1, y0, INK, 3))
  for i, x in enumerate(xs):
    o.append(T(sx(i), y0 + 34, x, 22, fill=MUTED, anchor="middle"))
  o.append(T((x0 + x1) / 2, y0 + 72, xlabel, 21, 600, MUTED, "middle"))
  for lab, col, vals in series:
    pts = " ".join(f"{sx(i):.1f},{sy(v):.1f}" for i, v in enumerate(vals))
    o.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="5" stroke-linejoin="round" stroke-linecap="round"/>')
    for i, v in enumerate(vals):
      o.append(C(sx(i), sy(v), 8, col, PAPER, 3))
    o.append(T(sx(len(xs) - 1) + 22, sy(vals[-1]) + 8, lab, 26, 800, col))
    o.append(T(sx(0) + 12, sy(vals[0]) - 18, f"{vals[0]:.1f}", 20, 700, col, "start"))
    o.append(T(sx(len(xs) - 1), sy(vals[-1]) - 18, f"{vals[-1]:.1f}", 20, 700, col, "middle"))
  return svg(w, h, "".join(o))


def funnel(stages, w, h, num=36, lab=25):
  o = []
  vmax = stages[0][1]
  step = h / len(stages)
  bh = step * 0.64
  for i, (name, v, col) in enumerate(stages):
    y = i * step + (step - bh) / 2
    ln = (w - 380) * v / vmax
    o.append(R(0, y, ln, bh, col, 12))
    o.append(T(num * 0.66, y + bh / 2 + num * 0.36, f"{v:,}", num, 800, ON.get(col, INK)))
    o.append(T(ln + 20, y + bh / 2 + lab * 0.36, name, lab, 600))
  return svg(w, h, "".join(o))


def stacked(rows, stages, w, h, label_w=0):
  o = []
  vmax = max(sum(r[1].get(k, 0) for k, _, _ in stages) for r in rows)
  step = (h - 60) / len(rows)
  for i, (lab, vals) in enumerate(rows):
    y = i * step
    if label_w:
      o.append(T(0, y + 58, lab, 28, 800))
    x = label_w
    for k, _, col in stages:
      v = vals.get(k, 0)
      if not v:
        continue
      ln = (w - label_w) * v / vmax
      o.append(R(x, y + 6, ln - 6, step - 16, col, 10))
      if ln > 90:
        o.append(T(x + ln / 2, y + step / 2 + 6, f"{v:.1f} h", 26, 800, ON.get(col, INK), "middle"))
      x += ln
  lx = label_w
  for k, lab, col in stages:
    o.append(R(lx, h - 34, 26, 26, col, 6))
    o.append(T(lx + 38, h - 12, lab, 24, fill=MUTED))
    lx += 90 + 13 * len(lab)
  return svg(w, h, "".join(o))


def chips(n, cols, size, gap, color, label):
  rows = (n + cols - 1) // cols
  w = cols * size + (cols - 1) * gap
  h = rows * size + (rows - 1) * gap + 56
  o = []
  for i in range(n):
    x = (i % cols) * (size + gap)
    y = (i // cols) * (size + gap)
    o.append(R(x, y, size, size, color, 16))
    o.append(R(x + size * 0.28, y + size * 0.28, size * 0.44, size * 0.44, "#ffffff", 6))
  o.append(T(w / 2, h - 10, label, 26, 700, INK, "middle"))
  return svg(w, h, "".join(o), bare=True)


def flow_svg(steps, w, h):
  """Numbered stages with arrows; steps = [(name, line1, line2, color)]."""
  o = []
  n = len(steps)
  gap = 46
  bw = (w - gap * (n - 1)) / n
  for i, (name, l1, l2, col) in enumerate(steps):
    x = i * (bw + gap)
    o.append(R(x, 4, bw, h - 8, CARD, 22, RULE, 2))
    o.append(R(x + 24, 4, bw - 48, 10, col, 5))
    o.append(C(x + 48, 76, 26, col))
    o.append(T(x + 48, 86, str(i + 1), 28, 800, ON.get(col, INK), "middle"))
    o.append(T(x + 88, 88, name, 28, 800))
    o.append(T(x + 26, 152, l1, 22, fill=MUTED))
    o.append(T(x + 26, 186, l2, 22, fill=MUTED))
    if i < n - 1:
      o.append(arrow(x + bw + 5, h / 2, x + bw + gap - 5))
  return svg(w, h, "".join(o), bare=True)


# ------------------------------------------------------------------ the talk

def dumbbell(rows, w, h, label_w=300):
  """Base-to-teacher gap per task; rows = [(task, sub, base, teacher, color)]."""
  o = []
  x0, x1 = label_w, w - 70
  sx = lambda v: x0 + (x1 - x0) * v / 100  # noqa: E731
  step = (h - 60) / len(rows)
  for v in (0, 25, 50, 75, 100):
    o.append(Lf(sx(v), 0, sx(v), h - 60, RULE, 2))
    o.append(T(sx(v), h - 22, f"{v}%", 20, fill=MUTED, anchor="middle"))
  for i, (task, sub, base, teacher, col) in enumerate(rows):
    y = i * step + step / 2
    o.append(T(0, y - 4, task, 28, 800))
    o.append(T(0, y + 26, f"{sub} · gap {teacher - base:.1f}", 21, fill=MUTED))
    o.append(Lf(sx(base), y, sx(teacher), y, GRAY, 12))
    o.append(C(sx(base), y, 18, GRAY, PAPER, 4))
    o.append(C(sx(teacher), y, 18, INK, PAPER, 4))
    o.append(T(sx(base), y + 52, f"{base:.1f}%", 24, 800, INK, "middle"))
    o.append(T(sx(teacher), y - 32, f"teacher {teacher:.1f}%", 22, 700, INK, "middle"))
  return svg(w, h, "".join(o))


def task_card(name, sub, accent, rows, cls=""):
  body = "".join(f'<div class="task-r"><span>{esc(k)}</span><div>{v}</div></div>' for k, v in rows)
  return f'<div class="task {cls}" style="border-top:12px solid {accent}"><div class="task-h">{esc(name)}<span>{esc(sub)}</span></div>{body}</div>'


def outcome_mix(panels, w, h, compact=False):
  """100% stacked bars of per-question outcomes over 8 samples, one panel per task.
  panels = [(title, [(label, all_wrong, mixed, all_right)])]."""
  o = []
  gap = 40 if compact else 60
  ft, fv, fl, fg_, sp = (20, 16, 16, 16, 9) if compact else (28, 22, 21, 22, 12)
  pw = (w - gap * (len(panels) - 1)) / len(panels)
  top, bot = (34, h - 64) if compact else (56, h - 110)
  ph = bot - top
  for p, (title, cols) in enumerate(panels):
    x0 = p * (pw + gap)
    o.append(T(x0, ft + 2, title, ft, 800))
    slot = pw / len(cols)
    bw = slot * 0.72
    for i, (lab, wrong, mixed, right) in enumerate(cols):
      x = x0 + slot * i + (slot - bw) / 2
      y = bot
      edge = i in (0, len(cols) - 1)
      for v, col, fg in ((wrong, GRAY, INK), (mixed, SKY, INK), (right, CORAL, "#ffffff")):
        sh = ph * v / 100
        o.append(R(x, y - sh + 1.5, bw, sh - 3, col, 6))
        if edge:
          o.append(T(x + bw / 2, y - sh / 2 + fv * 0.36, f"{v:.0f}", fv, 800, fg, "middle"))
        y -= sh
      o.append(T(x + bw / 2, bot + fl + 8, lab, fl, 600, MUTED, "middle"))
  lx = 0
  for lab, col in (("all 8 answers right", CORAL), ("some right", SKY), ("all 8 wrong", GRAY)):
    o.append(R(lx, h - fg_ - 8, fg_ + 4, fg_ + 4, col, 6))
    o.append(T(lx + fg_ + 14, h - 8, lab, fg_, fill=MUTED))
    lx += 5 * sp + sp * len(lab)
  return svg(w, h, "".join(o))


def B(step, inner):
  """A build group: hidden when the slide opens, revealed by the `step`-th forward click."""
  return f'<div class="build" data-step="{step}">{inner}</div>'


def Bi(step, inner):
  """Like B, but step 0 is shown when the slide opens."""
  return B(step, inner) if step else inner


def merge_notes(*parts):
  """Join the notes of slides shown together, summing their timings."""
  total, out = 0.0, []
  for p in parts:
    m = re.search(r"\s*~(\d+(?:\.\d+)?) (min|s)\.\s*$", p)
    if m:
      total += float(m.group(1)) / (60 if m.group(2) == "s" else 1)
      p = p[:m.start()]
    out.append(p.strip())
  return "\n".join(out) + f" ~{total:g} min."


def head(col, span, top, text, sub=""):
  """A part header on a slide that covers two topics; appendix links can point at it."""
  s = f"<span>{esc(sub)}</span>" if sub else ""
  return box(col, span, top, 50, f'<div class="hh">{esc(text)}{s}</div>')


def vsep(top=TOP, bottom=BOT, col=6):
  """A vertical rule in the gutter before `col`, separating the two halves of a slide."""
  return f'<div class="vsep" style="left:{gx(col) - GUT / 2 - 1:.1f}px;top:{top:.1f}px;height:{bottom - top:.1f}px"></div>'


def srow(big, cap, accent=TEAL):
  """A compact stat: the number beside its caption."""
  return f'<div class="srow"><div class="srow-b" style="color:{accent}">{big}</div><div class="srow-c">{cap}</div></div>'


def runin(title, body, accent=None):
  """A one-paragraph card whose bold title runs into its text."""
  return f'<div class="ri" style="border-left:8px solid {accent or NEUTRAL}"><div><b>{title}</b> {body}</div></div>'


def step_head(i, top, title):
  names = ["Measure", "Generate", "Verify", "Distil", "Reinforce", "Serve"]
  c = STEP_COLOR[i]
  return box(0, 12, top, 46, f'<div class="sh" style="border-bottom-color:{c}"><b style="background:{c};color:{ON.get(c, INK)}">{i + 1}</b>'
                             f'<i>{names[i]}</i><span>{esc(title)}</span></div>')


def journey_pair(i, title, steps, notes, source=""):
  """Two consecutive training steps, one row each: header, visual (5 cols), three run-in panels (7 cols)."""
  rh = (BOT - TOP - 24) / 2
  inner = ""
  for k, (stitle, visual, did, decision, learned) in enumerate(steps):
    y = TOP + k * (rh + 24)
    top, h = y + 54, rh - 54
    ph = (h - 2 * 10) / 3
    panels = "".join(box(5, 7, top + j * (ph + 10), ph, f'<div class="pr"><div><span class="prl" style="color:{col}">{lab}</span>{txt}</div></div>')
                     for j, (lab, txt, col) in enumerate((("What we did", did, MUTED), ("Decision", decision, BLUE), ("What we learned", learned, DARK))))
    inner += Bi(2 * k, step_head(i + k, y, stitle) + box(0, 5, top, h, visual, "vis")) + B(2 * k + 1, panels)
  content(2, f"The journey · steps {i + 1}–{i + 2} of 6", title, inner, notes, source, also=[s[0] for s in steps])


# compact visuals for the journey rows: viewBox 640 x 264

VW, VH = 640, 264


def measure_vis():
  ax, pt = 440, 16
  o = [T(0, 22, "Is the grader right?", 21, 800), T(70, 54, "gold answers", 16, 700, MUTED), T(180, 54, "shuffled answers", 16, 700, MUTED)]
  for i, (k, g, s_) in enumerate([("BIRD", "100%*", "0.2%"), ("FinQA", "100%", "0.6%"), ("MATH", "100%", "0.0%")]):
    y = 88 + i * 32
    o += [T(0, y, k, 19, 800), T(70, y, g, 19, 800, BLUE), T(180, y, s_, 19, 800, BLUE)]
  o += [T(330, 22, "How much does serving noise", 21, 800), T(330, 46, "move the score?", 21, 800), Lf(ax, 62, ax, 160, INK, 2)]
  for i, (k, e, l, u) in enumerate([("BIRD", 0.3, -0.8, 1.4), ("FinQA", -1.9, -3.3, -0.5), ("MATH", -0.12, -0.88, 0.62)]):
    y = 80 + i * 32
    o += [T(330, y + 6, k, 17, 800), Lf(ax + l * pt, y, ax + u * pt, y, GRAY, 8), C(ax + e * pt, y, 9, INK, PAPER, 3),
          T(ax + 2 * pt + 14, y + 6, f"{e:+.1f} [{l:+.1f}, {u:+.1f}]", 16, 700)]
  o += [T(ax, 180, "0", 15, fill=MUTED, anchor="middle"), T(ax - 2 * pt, 180, "−2", 15, fill=MUTED, anchor="middle"),
        T(ax + 2 * pt, 180, "+2", 15, fill=MUTED, anchor="middle"),
        R(0, 194, VW, 70, PINK, 12), T(18, 220, "Teacher − untrained:", 17, 700, DARK),
        T(18, 250, "BIRD +30.1 · FinQA +31.7 · MATH +17.6 points", 21, 800, DARK)]
  return svg(VW, VH, "".join(o))


def generate_vis():
  def row(y, who, sub, fill, fg, mult, head_, l1, l2, tag, tagc):
    return [R(0, y, 150, 124, fill, 16), T(75, y + 58, who, 24, 800, fg, "middle"), T(75, y + 88, sub, 17, 400, fg, "middle"),
            arrow(160, y + 62, 224, MUTED, 4), T(192, y + 46, mult, 18, 800, MUTED, "middle"),
            R(232, y, 408, 124, CARD, 16, RULE, 2), T(250, y + 30, head_, 20, 800), T(250, y + 58, l1, 16, 600, INK),
            T(250, y + 84, l2, 16, 400, MUTED), T(250, y + 110, tag, 17, 700, tagc)]
  o = (row(4, "Teacher", "8× H200", INK, "#ffffff", "× 4", "Teacher answers, T 0.7", "BIRD 33,728 · FinQA 24,436 · MATH 25,048",
           "26–38 minutes per task on 8 GPUs", "for a weak student", TEAL) +
       row(140, "Student", "1 v6e chip each", GOLD, INK, "× 8", "Its own answers, T 0.7", "MATH 50,096 · the control on every task",
           "no teacher time", "for a strong student", GOLD))
  return svg(VW, VH, "".join(o))


def verify_vis():
  stages = [("teacher answers", 33728, INK), ("pass the grader", 20133, BLUE), ("distinct, ≤ 4 per question", 10504, SKY), ("BIRD training set", 10112, TEAL)]
  o = []
  for i, (name, v, col) in enumerate(stages):
    y = i * 34
    ln = 300 * v / stages[0][1]
    o += [R(0, y, ln, 26, col, 8), T(12, y + 19, f"{v:,}", 17, 800, ON.get(col, INK)), T(ln + 14, y + 19, name, 17, 600)]
  xs = (0, 220, 340, 500)
  o += [T(x, 162, h_, 14, 800, MUTED, extra='letter-spacing="1.5"') for x, h_ in zip(xs, ("TASK", "ANSWERS", "PASS THE GRADER", "TRAINING SET"))]
  o.append(Lf(0, 174, VW, 174, INK, 2))
  for r, row in enumerate([["FinQA, teacher", "24,436", "18,487", "6,016"], ["MATH, student", "50,096", "72.6%", "4,992"]]):
    y = 206 + r * 40
    o += [T(x, y, c, 18) for x, c in zip(xs, row)]
    if r == 0:
      o.append(Lf(0, y + 14, VW, y + 14, RULE, 2))
  return svg(VW, VH, "".join(o))


def distil_vis():
  o = [T(0, 22, "LoRA fine-tuning on TPU v6e-8", 21, 800),
       *[R(i * 56, 34, 44, 44, DARK, 10) for i in range(8)], *[R(i * 56 + 13, 47, 18, 18, "#ffffff", 4) for i in range(8)],
       T(0, 104, "117 / 69 / 57 steps (BIRD / FinQA / MATH) · 10–27 minutes", 17, 600, MUTED),
       T(0, 136, "Checks we ran before trusting any result", 20, 800),
       R(0, 148, 310, 64, CARD, 12, RULE, 2), T(16, 180, "0.998", 26, 800, TEAL), T(16, 202, "trainer and server agree (top-1)", 15, fill=MUTED),
       R(330, 148, 310, 64, CARD, 12, RULE, 2), T(346, 180, "1 : 1", 26, 800, TEAL), T(346, 202, "retrain gave a byte-identical model", 15, fill=MUTED),
       T(0, 236, "rank 256 · alpha 512 · attention + MLP · batch 256 · 3 epochs", 17, fill=INK),
       T(0, 260, "Qwen3 and Gemma 4 loaders · loss on answer tokens only", 17, fill=INK)]
  return svg(VW, VH, "".join(o))


def build():
  # ---------------- opening
  add(f'''<div class="kicker" style="left:{ML}px;top:150px;color:{GOLD}"><span class="dot" style="background:{GOLD}"></span>TPU-Distil · October 2026</div>
<h1 class="hero" style="left:{ML}px;top:320px;width:1100px">Big-model quality<br>at a small-model price</h1>
<div class="div-sub" style="left:{ML}px;top:580px;width:980px">How we taught small models to do the job of a 552B model on SQL, finance and maths, and served each one on a single Cloud TPU chip</div>
<div class="author" style="left:{ML}px;top:790px">Jawad Amin and Fred<br><span style="color:{SKY}">github.com/jawadaminGOOG/tpu-distil</span></div>
<div class="circle" style="left:1250px;top:120px;width:560px;height:560px;background:{TEAL}"></div>
<div class="circle" style="left:1140px;top:560px;width:250px;height:250px;background:{GOLD}"></div>
<div class="circle" style="left:1620px;top:640px;width:170px;height:170px;background:{CORAL}"></div>
<div class="circle" style="left:1440px;top:820px;width:100px;height:100px;background:{SKY}"></div>''',
      """Welcome. This talk is about getting the quality of a big model at the price of a small one.
A frontier model does specialist work well, but it needs a whole 8-GPU server. A small model is cheap, but it isn't good at the job. We used the big model to teach small models on three tasks: SQL, financial questions and competition maths. Then we served each student on a single TPU chip. Along the way we found that the recipe has to change when the student is already strong.
After that, Fred will show his experiment, which took a different route: teaching models to draw. ~1 min.""", "dark")

  agenda = []
  pw = (CW - 4 * 32) / 5
  items = [("The problem", "Why big models are hard to serve and small models fall short", 6),
           ("The experiments", "The three tasks, the two students and how we chose them", 5),
           ("The journey", "Our software stack and the six training steps, with what we learned at each one", 11),
           ("What we got", "The results, the two recipes and why we trust the numbers", 11),
           ("Fred's experiment", "How Fred taught models to draw", 7)]
  for i, (t, d, mins) in enumerate(items):
    x = ML + i * (pw + 32)
    num_col = DARK if CH_COLOR[i] == SKY else CH_COLOR[i]   # lavender is too pale for text on white
    agenda.append(f'''<div class="agenda" style="left:{x:.1f}px;top:{TOP}px;width:{pw:.1f}px;height:{BOT - TOP}px;border-top:14px solid {CH_COLOR[i]}">
<div class="ag-n" style="color:{num_col}">{i + 1:02d}</div><div class="ag-t">{esc(t)}</div><div class="ag-d">{esc(d)}</div><div class="ag-m">~{mins} min</div></div>''')
  add(f'''<div class="kicker" style="left:{ML}px;top:70px"><span class="dot" style="background:{CORAL}"></span>Agenda</div>
<h2 class="title" style="left:{ML}px;top:104px;width:{CW}px">What we'll cover</h2>{"".join(agenda)}''',
      """The talk has five parts. First the problem. Then the three experiments and how we set them up. Then the training process, step by step. Then the results, including the point where the recipe had to change. And then Fred's experiment. At the end I'll compare our work with Meta's TorchTitan RL work, give you a recipe you can try yourself, and sum up. ~30 s.""")

  # ---------------- 1. problem
  dark_divider(1, "The problem", "Frontier models do the job well, but they are expensive and GPUs are hard to get. Small models are cheap, but they don't know the domain.", CORAL,
               "Part 1: the problem. ~6 minutes.")

  y1 = TOP + 62
  hw_svg = svg(820, 230, "".join([
      *[R(20 + (i % 4) * 84, 6 + (i // 4) * 84, 70, 70, INK, 12) for i in range(8)],
      *[R(20 + (i % 4) * 84 + 20, 6 + (i // 4) * 84 + 20, 30, 30, "#ffffff", 5) for i in range(8)],
      T(181, 214, "8× H200 for one teacher replica", 22, 700, INK, "middle"),
      T(450, 112, "vs", 64, 700, CORAL, "middle", 'font-family="Inter Tight, Figtree, Arial, sans-serif"'),
      R(560, 4, 160, 160, DARK, 20), R(605, 49, 70, 70, "#ffffff", 10),
      T(640, 214, "1× TPU v6e chip for the student", 22, 700, INK, "middle")]), bare=True)
  sh = (BOT - y1 - 230 - 12 - 2 * 12) / 3
  ch = (BOT - y1 - 440 - 16)
  content(0, "The problem", "Big models are expensive; small ones fall short",
          head(0, 6, TOP, "One frontier model needs a whole GPU server") +
          box(0, 6, y1, 230, hw_svg) +
          "".join(box(0, 6, y1 + 242 + i * (sh + 12), sh, srow(b, c, a)) for i, (b, c, a) in enumerate([
              ("552B", "parameters in the teacher, DeepSeek-V4.1-Flash", INK),
              ("$36.60 / h", "for one teacher replica, at the cheapest listed H200 rate", INK),
              ("$2.70 / h", "for one TPU v6e chip, on demand", DARK)])) +
          B(1, vsep() +
          head(6, 6, TOP, "Every correct answer has a price") +
          box(6, 6, y1, 440, quadrant([("Teacher", 1.438, 60.7, INK, "end"), ("Small model, untrained", 0.178, 30.6, GRAY, "start")], 800, 430))) +
          B(2, box(6, 3, y1 + 456, ch, card("The big model is accurate but expensive", "It writes correct SQL for <b>60.7%</b> of questions, and 1,000 correct answers cost <b>$1.44</b>.", INK, cls="sm")) +
          box(9, 3, y1 + 456, ch, card("The small model is cheap but often wrong", "1,000 correct answers cost only <b>$0.18</b>, but it gets just <b>30.6%</b> of questions right.", GRAY, cls="sm"))),
          merge_notes("""Let's start with the hardware. Our teacher, DeepSeek-V4.1-Flash, is a mixture-of-experts model with 552 billion parameters. One copy of it fills a whole server of eight H200 GPUs, which costs 36 dollars 60 an hour at the cheapest listed rate.
GPUs like these are hard to get. Each copy of the model ties up a full server, so serving more users means renting more servers. A single TPU v6e chip costs 2 dollars 70 an hour. The question is whether we can move useful work from the left side of this slide to the right. ~1.5 min.""",
                      """We put a price on every correct answer. On the BIRD text-to-SQL benchmark, the teacher gets 60.7 percent of questions right, and a thousand correct answers cost a dollar 44. The untrained small model on one TPU chip costs 18 cents per thousand correct answers, but it gets less than a third of the questions right.
We don't want either of these. We want a model in the top-left corner, one that is cheap and accurate. Keep this chart in mind, because we'll come back to it. ~1.5 min."""),
          "BIRD dev: 1,534 questions on 11 databases never used in training. Cost = $ per hour ÷ (requests per second × 3,600 × accuracy).",
          also=["One frontier model needs a whole GPU server", "Every correct answer has a price"])

  bet_h = 196
  top_h = BOT - TOP - bet_h - 20
  content(0, "The problem", "Generic small models miss domain details",
          box(0, 6, TOP, 112, '<div class="q sm"><span>Question</span>Which county is member Adela O\'Gallagher\'s hometown in?</div>') +
          box(0, 6, TOP + 124, (top_h - 136) / 2, code("WHERE member.first_name = 'Adela'\n  AND member.last_name = 'O\\'Gallagher'", GRAY, "Untrained · SQLite syntax error", cls="sm")) +
          B(1, box(0, 6, TOP + 136 + (top_h - 136) / 2, (top_h - 136) / 2, code("WHERE m.first_name = 'Adela'\n  AND m.last_name = 'O''Gallagher'", CORAL, "Trained student · correct rows", cls="sm"))) +
          B(2, box(6, 6, TOP, 224, hbars([("Simple", "925 questions", 40.3, GRAY), ("Moderate", "465 questions", 20.3, GRAY),
                                     ("Challenging", "144 questions", 10.3, GRAY)], 760, 224, 75, 250, bar_h=44)) +
          box(6, 6, TOP + 236, top_h - 236, learn("The untrained 1.7B model gets only <b>10%</b> of the hardest questions right. Of its wrong answers, <b>574</b> return the wrong rows and <b>462</b> are not valid SQLite, because the model invents columns or uses another dialect's syntax. Companies see the same problem with their own schemas and SQL dialects.", CORAL, "Why fine-tune for the domain", cls="sm"))) +
          B(3, box(0, 12, BOT - bet_h, bet_h, f'''<div class="bet"><div class="bet-k" style="color:{DARK}"><span class="dot" style="background:{GOLD}"></span>The bet</div>
<div class="bet-s">We bet that a big model could teach a small one, and that the small one could run on a single TPU chip.</div>
<div class="bet-b">Before we started, we set two targets: at least <b>5 points</b> better than the untrained student, and at least <b>10×</b> cheaper per correct answer than the teacher.</div></div>''')),
          merge_notes("""Here is what "weak in the domain" looks like, with a real BIRD question. The untrained small model escapes the apostrophe the way MySQL does, and SQLite rejects the query. After training, the student uses SQLite's doubled quote and gets the right rows.
On the right you can see that the untrained model solves 40 percent of the simple questions and only 10 percent of the hardest ones. 462 of its wrong answers aren't even valid SQL. This is the gap companies care about: their own schemas and their own SQL dialect. ~1.5 min.""",
                      """Here was our bet. We use the big model as a teacher, train the small model on its answers, and serve the student on a single TPU chip.
Before we started, we wrote down what success would mean: the student had to be at least five points better than before, with a confidence interval above zero, and at least ten times cheaper per correct answer than the teacher. We tested the bet three times, and the third test showed us where it breaks down. ~1 min."""),
          "BIRD dev; accuracy by BIRD's difficulty label for the untrained Qwen3-1.7B; example dev-1347, question paraphrased")

  # ---------------- 2. experiments
  dark_divider(2, "The experiments", "We ran the same pipeline on three tasks with two different students.", GOLD, "Part 2: the experiment setup. ~5 minutes.")

  th = 500
  band = BOT - TOP - th - 12 - 58
  content(1, "The experiments", "Each task has an automatic grader",
          box(0, 4, TOP, th, task_card("BIRD", "text-to-SQL", BLUE, [
              ("Input", "a database schema, example rows, a hint and a question"),
              ("Answer", "one SQLite query"),
              ("Grader", "runs the query and compares its rows with the gold rows"),
              ("Evaluate", "dev: 1,534 questions on 11 unseen databases"),
              ("Train", "8,432 questions on 64 databases"),
              ("Student", "<b>Qwen3-1.7B</b> scores 30.6%; the teacher scores 60.7%")], cls="sm")) +
          box(4, 4, TOP, th, task_card("FinQA", "financial reports", SKY, [
              ("Input", "one earnings-report page: text, a table, a question"),
              ("Answer", "<span class=mono>Answer: &lt;number&gt;</span>"),
              ("Grader", "checks that the number is within 1% of the gold answer"),
              ("Evaluate", "test: 1,147 questions on unseen report pages"),
              ("Train", "6,109 questions after removing overlaps"),
              ("Student", "<b>Qwen3-1.7B</b> scores 44.0%; the teacher scores 75.7%")], cls="sm")) +
          box(8, 4, TOP, th, task_card("MATH", "competition maths", PINK, [
              ("Input", "a competition problem, 7 subjects, 5 levels"),
              ("Answer", "<span class=mono>\\boxed{answer}</span>"),
              ("Grader", "checks the boxed answer against the gold answer after standard clean-up"),
              ("Evaluate", "test: 5,000 problems"),
              ("Train", "6,262 problems after removing overlaps"),
              ("Student", "<b>Gemma 4 E2B</b> scores 74.9%; the teacher scores 92.5%")], cls="sm")) +
          B(1, head(0, 12, TOP + th + 12, "One student starts weak, the other starts strong", "untrained student vs teacher, same prompt") +
          box(0, 8, BOT - band, band, card("Qwen3-1.7B starts weak", "It is about 30 points behind the teacher on SQL and finance, so it has the most to learn. It is also the cheapest to serve.", TEAL, cls="sm")) +
          box(8, 4, BOT - band, band, card("Gemma 4 E2B starts strong", "It already solves 74.9% of MATH problems. We wanted to know if a teacher still helps.", GOLD, cls="sm"))),
          merge_notes("""We ran three experiments through the same pipeline. In BIRD, the model writes SQL against a database schema, and the grader runs the query and compares the rows. In FinQA, the model answers a numerical question about a page from an earnings report, which has text and a table, and the grader checks that the number is within one percent. In MATH, the model solves competition problems and puts its answer in a box, and the grader checks for an exact match.
Each task is one module in the code, with its own prompt, answer extraction and grader. Everything else is shared: generation, verification, training and serving. Every grader also passes two checks: gold answers score 100 percent and shuffled answers score close to zero. ~1.5 min.""",
                      """We chose each student by how far it was behind the teacher, measured on the same benchmark with the same prompt. Qwen3-1.7B is about 30 points behind on both SQL and finance, so it has a lot to learn. The next size up, Qwen3-4B, already matched the teacher on BIRD, so there was nothing left to teach it.
For the third experiment we picked a different kind of student on purpose: Gemma 4 E2B on competition maths. It is already strong, at 74.9 percent, and much closer to the teacher. That let us test whether a teacher still helps a student like that. ~1.5 min."""),
          ["Teacher DeepSeek-V4.1-Flash on 8× H200 for all three; students served by vLLM-TPU on one v6e chip; non-thinking mode throughout",
           "Untrained student vs teacher, same prompt, greedy decoding; Qwen3-4B already scored 61.7% on BIRD and was not used"],
          also=["One student starts weak, the other starts strong"])

  hcard = (BOT - TOP - 30) / 2
  dec = [("Tasks", "Tasks with automatic graders", "A program checks every answer, so one run takes minutes and can show a 2-point change. An agentic benchmark needs hours per run.", CORAL),
         ("Students", "A weak student and a strong one", "With two students we can see when a teacher helps, and when it doesn't.", GOLD),
         ("Training data", "Only verified answers", "The grader checks every training answer, because a wrong answer would teach the student the wrong thing.", TEAL),
         ("Format", "Short answers", "Teacher and student both reason briefly. Without long reasoning traces, the teacher samples 4× faster.", BLUE),
         ("Training", "LoRA on one TPU host", "LoRA fits on a single TPU v6e-8, and every size is a multiple of 256 to match the chip's matrix unit.", SKY),
         ("Evaluation", "Comparisons within one run", "We evaluate every model, a control and a second copy of the baseline together.", PINK)]
  cards = []
  for i, (k, head_, why, col) in enumerate(dec):
    c, r = (i % 3) * 4, i // 3
    cards.append(box(c, 4, TOP + r * (hcard + 30), hcard,
                     f'''<div class="dec" style="border-top:10px solid {col}"><div class="dec-k">{esc(k)}</div>
<div class="dec-c">{esc(head_)}</div><div class="dec-w">{esc(why)}</div></div>'''))
  content(1, "The experiments", "Six decisions shaped the experiments", "".join(Bi(i, c) for i, c in enumerate(cards)),
          """Six decisions shaped all three experiments. We picked tasks with automatic graders, so a single run takes a few minutes and can show a two-point change. We used a weak student and a strong one, to see when a teacher helps. We trained only on verified answers. We kept answers short, without long reasoning traces. We trained with LoRA on one TPU host. And we compared models only within the same run, alongside a control and a second copy of the baseline. ~1.5 min.""")

  # ---------------- 3. journey
  dark_divider(3, "The journey", "The software we used, the six training steps, and what we learned at each step.", TEAL, "Part 3: the stack and the training journey. ~11 minutes.")

  steps = [(n, a, b, STEP_COLOR[i]) for i, (n, a, b) in enumerate([
      ("Measure", "graders, noise floor", "teacher vs student"), ("Generate", "teacher answers, or", "the student's own"),
      ("Verify", "grade every answer", "audit for leaks"), ("Distil", "LoRA SFT on TPU", "10–27 minutes"),
      ("Reinforce", "GRPO, 6 rounds", "rewards correct answers"), ("Serve", "1 chip per replica", "price each answer")])]
  layers = [("Teacher", "8× H200, Vertex AI", INK, ["DeepSeek-V4.1-Flash", "Model Garden vLLM", "TP 8 + expert parallel"]),
            ("Pipeline", "CPU VM", GOLD, ["Python standard library", "sqlite3", "JSONL records", "paired bootstrap"]),
            ("Student serving", "1 TPU v6e chip each", TEAL, ["vLLM-TPU", "tpu-inference backend", "JAX"]),
            ("Training", "TPU v6e-8 host", BLUE, ["JAX + libtpu", "Tunix", "qwix LoRA", "Flax NNX", "Optax", "Orbax"]),
            ("Glue", "everywhere", SKY, ["Hugging Face transformers", "safetensors", "Cloud Storage"])]
  fh = 200
  st = TOP + fh + 16 + 58
  lh = (BOT - st - 4 * 10) / 5
  stack_html = "".join(box(0, 8, st + i * (lh + 10), lh,
                           f'<div class="layer sm" style="border-left-color:{c}"><div class="layer-l">{esc(n)}<span>{esc(where)}</span></div>'
                           f'<div class="pills">{"".join(f"<span class=pill>{esc(p)}</span>" for p in ps)}</div></div>')
                       for i, (n, where, c, ps) in enumerate(layers))
  rc = [("One code path", "Both students share one code path; only the task module changes."),
        ("One checkpoint format", "Trainer, server and storage all use merged Hugging Face safetensors."),
        ("Simple pipeline scripts", "It uses standard-library Python and JSONL, so every number can be regenerated."),
        ("One trainer for both models", "The same Tunix trainer handles Qwen3 and Gemma 4 on one v6e-8.")]
  rh = (BOT - st - 3 * 10) / 4
  content(2, "The journey", "Every experiment follows the same six steps",
          box(0, 12, TOP, fh, flow_svg(steps, CW, 250)) +
          B(1, head(0, 12, TOP + fh + 16, "We built everything on open-source libraries", "distilling takes 10–27 min on one TPU v6e-8 host") +
          stack_html +
          "".join(box(8, 4, st + i * (rh + 10), rh, runin(t, b)) for i, (t, b) in enumerate(rc))),
          merge_notes("""Here is the whole process on one slide. All three experiments follow it. We measure the gap and the noise. We generate training answers, from the teacher for a weak student or from the student itself for a strong one. We check every answer with the task's grader. We distil with LoRA on one TPU host. We reinforce with GRPO. And we serve each student on one chip and work out the price of each answer. Distillation takes 10 to 27 minutes, and the final students are 4.8 to 17.5 times cheaper per correct answer than the teacher. ~1 min.""",
                      """Everything in the stack is open source. The teacher runs in Vertex AI Model Garden's vLLM container on eight H200s. The pipeline is plain Python standard-library scripts that write JSONL, so we can regenerate every number from the raw files. We serve students with vLLM-TPU, one copy per chip, and train them with Tunix and qwix LoRA in JAX on one v6e-8 host. The same trainer handles both Qwen3 and Gemma 4. The trainer, the server and storage all use one checkpoint format, Hugging Face safetensors. ~1 min."""),
          "Versions: vLLM 0.26.1rc1 TPU build, JAX 0.11.0, libtpu 0.0.44, Tunix 0.1.8, qwix 0.1.8, Flax 0.12.8, transformers 5.12.1 (docs/RECIPE.md)",
          also=["We built everything on open-source libraries"])

  journey_pair(0, "Measure the noise, then choose the training answers", [
      ("Check the measurements before the experiment", measure_vis(),
       "We checked that every grader gives gold answers 100% and shuffled answers close to 0%. We also served two copies of each untrained model to measure noise.",
       "We compare models only <b>within a single run</b>, and every run includes a second copy of the baseline.",
       "Greedy decoding on vLLM-TPU isn't perfectly repeatable. Two copies of the same FinQA model differed by 1.9 points, so we can't trust a small gain without a noise floor."),
      ("Choose who writes the training answers", generate_vis(),
       "We sampled the teacher 4 times and the student 8 times per question, using training questions only. Both write the same short format: brief reasoning, then one answer.",
       "A weak student learns from the teacher's answers. A strong student learns from <b>its own verified answers</b>, which keep its style of reasoning.",
       "The training answers should look like what the student will write when it serves. On MATH, the teacher's shorter solutions made a strong student worse.")],
      merge_notes("""Step one is to check the measurements before running the experiment. Every grader passes two checks: gold answers score 100 percent and shuffled answers score almost zero. Then we serve two copies of each untrained model side by side. On FinQA the two copies differ by 1.9 points, because greedy decoding on TPU serving isn't perfectly repeatable. So every comparison in this talk comes from a single run that includes a second copy of the baseline. The gaps to the teacher are 30, 32 and 18 points. ~1.5 min.""",
                  """Step two is choosing who writes the training answers. For a weak student, that's the teacher. We sample it four times per question at temperature 0.7, which gives 25 to 34 thousand answers per task in under 40 minutes. For a strong student, it's the student itself. We sample it eight times per question, and we don't need the teacher at all. We also train on the student's own answers as a control on every task, and on MATH that control won. Both teacher and student write the same short format: brief reasoning, then one answer. ~1.5 min."""),
      "* BIRD gold scores 100% of the 1,530 gold queries that execute (99.7% of 1,534)")

  journey_pair(2, "Verify every answer, then distil on one TPU host", [
      ("Keep only the answers the grader accepts", verify_vis(),
       "We graded every answer and kept only the correct ones, with up to four distinct answers per question. A separate audit re-checked every row and looked for overlap with the evaluation sets.",
       "We train <b>only on verified answers</b>, and we reject any row that comes from an evaluation database, report page or problem.",
       "The grader turned out to be the most valuable code we wrote, because 40% of the teacher's BIRD answers were wrong. The audit found <b>no leaks</b> on any task."),
      ("Distil the student on one TPU host", distil_vis(),
       "We trained low-rank adapters on all eight chips of one TPU v6e-8 host, merged them into the model and exported a standard checkpoint for serving. Every task used the same settings.",
       "We used LoRA instead of full fine-tuning because it fits on one host. Every size is a multiple of 256 to match the TPU's matrix unit.",
       "Before you trust a result, check that the trainer and the server compute the same model. Keep adapter weights in float32, because in bf16 the optimizer's first update divides zero by zero.")],
      merge_notes("""Step three is verification. Every answer goes through the task's grader, and we keep it only if it's correct. That threw out 40 percent of the teacher's BIRD answers and a quarter of its FinQA answers. The teacher is good, but it isn't perfect. On MATH we filter the student's own answers the same way. A separate audit re-grades every row we kept and checks for any overlap with the evaluation sets. It found no leaks on any of the three tasks. ~1.5 min.""",
                  """Step four is distillation. We train low-rank adapters on all eight chips of one TPU host, with the same settings on every task. That's 117, 69 and 57 optimizer steps, and 10 to 27 minutes of training. Before trusting any result, we checked that the training model and the serving model agree: they pick the same top token 99.8 percent of the time on Qwen3 and 99.4 percent on Gemma. A full retrain also reproduced a byte-identical checkpoint. One lesson we learned the hard way: adapter weights have to stay in float32. ~1.5 min."""),
      "Step 3 bars: BIRD; table: FinQA and MATH (docs/RECIPE.md)")

  rounds = ["KD", "1", "2", "3", "4", "5"]
  bird_mix = [(r, 100 - p8, mx, p8 - mx) for r, p8, mx in zip(rounds, (73.09, 73.11, 73.52, 73.90, 74.12, 74.34), (34.17, 32.8, 31.0, 30.2, 26.7, 27.1))]
  finqa_mix = [(r, w_, mx, rt) for r, w_, mx, rt in zip(rounds, (14.62, 14.29, 14.00, 13.70, 13.37, 13.03), (42.05, 38.48, 35.18, 31.28, 27.91, 25.96),
                                                      (43.33, 47.23, 50.83, 55.02, 58.72, 61.01))]
  journey_pair(4, "Reinforce with GRPO, then serve on a single chip", [
      ("RL makes sometimes-right answers always right", outcome_mix([("BIRD", bird_mix), ("FinQA", finqa_mix)], VW, VH, compact=True),
       "We ran six rounds of GRPO. The student answers each training question 8 times, gets a reward of 1 when the grader accepts an answer, and moves toward the better answers in each group.",
       "The update learns only from questions with <b>mixed outcomes</b>, because all-right and all-wrong groups give it nothing to compare. A small KL penalty keeps the student close to where it started.",
       "On FinQA, the share of questions the student always gets right grew from 43% to 61%, while the share it never gets right barely moved. On held-out data, RL added 4.0 points on BIRD, 5.8 on FinQA and 1.8 on MATH."),
      ("Each student runs on a single chip",
       hbars([("Teacher", "8× H200", 1.438, INK), ("Untrained", "1 TPU v6e chip", 0.178, GRAY), ("Trained student", "1 TPU v6e chip", 0.091, CORAL)],
             VW, VH, 1.6, 240, value_fmt="${:.3f}", bar_h=48),
       "We served every model with the same flags, one vLLM-TPU copy per chip, and raised the load until errors appeared. Then we priced the best error-free rate.",
       "We measure the <b>cost of each correct answer</b>, because both accuracy and throughput affect it. Tokens per second would ignore accuracy.",
       "Answer length drives throughput. The trained BIRD student is faster than the untrained one, at 15.8 requests per second, but the MATH student's long answers make it slower than the teacher.")],
      merge_notes("""Step five is reinforcement learning. GRPO samples the student eight times per training question. An answer gets a reward of one if the grader accepts it, and the update pushes the student toward the better answers in each group. Only questions with mixed outcomes teach it anything. If all eight answers are right, or all eight are wrong, there's nothing to compare.
The bars sort every training question, in every round, into three groups. Teal means all eight answers were right, gold means some were right, and gray means all eight were wrong. On FinQA the teal block grows from 43 to 61 percent, and it grows by taking over the gold block. The gray block barely changes. So RL takes answers the student can sometimes reach and makes it give them every time. It doesn't teach the student questions it never solves; that's what distillation is for. On held-out data, RL is worth 4 points on BIRD, 5.8 on FinQA and 1.8 on MATH. ~2 min.""",
                  """Step six is serving, and putting a price on each answer. Every model gets the same serving flags and one TPU chip per copy. We raise the load until the server starts returning errors, and we price the best rate it reached without errors. We measure cost per correct answer, because a fast wrong answer is worth nothing. On BIRD, a thousand correct answers from the trained student cost 9 cents. That's half the cost of the untrained model and a sixteenth of the teacher's. Answer length matters a lot here, as we'll see with MATH. ~1 min."""),
      ["Step 5: training pool, 8 samples per question at T 1.0, policy that produced each round's samples; BIRD all-wrong = 100 − pass@8, all-right = pass@8 − mixed",
       "Step 6: serving cost per 1,000 correct BIRD answers; student at the on-demand TPU v6e rate, teacher at the cheapest listed H200 rate"])

  # ---------------- 4. results
  dark_divider(4, "What we got", "What each experiment achieved, why the recipe changed for the strong student, and why we trust the numbers.", BLUE, "Part 4: results. ~11 minutes.")

  def lhead(name, sub):
    return f'<div class="lh">{esc(name)}<span>{esc(sub)}</span></div>'
  lad_h = 392
  sy0 = TOP + 60 + lad_h + 14
  srh = (BOT - sy0 - 2 * 10) / 3
  cols = [("BIRD", "SQL · +21.6 points", [("base", "Base", 30.6), ("kd", "KD", 48.2), ("rl", "+ GRPO", 52.2)], 60.7,
           [("+17.6", "points from distillation, 95% CI [+15.2, +20.1]", TEAL), ("+4.0", "more points from GRPO, 95% CI [+2.2, +5.9]", CORAL),
            ("72%", "of the gap to the teacher closed, for $136 of compute", CORAL)]),
          ("FinQA", "finance · +28.4 points", [("base", "Base", 44.0), ("kd", "KD", 66.6), ("rl", "+ GRPO", 72.5)], 75.7,
           [("+22.6", "points from distillation, 95% CI [+19.4, +25.7]", TEAL), ("+5.8", "more points from GRPO, 95% CI [+3.9, +7.8]", CORAL),
            ("90%", "of the gap to the teacher closed, for $79 of compute", CORAL)]),
          ("MATH", "Gemma · +3.5 points", [("base", "Base", 74.9), ("self", "Self", 76.6), ("rl", "+ GRPO", 78.4)], 92.5,
           [("+3.5", "points from self-distillation and GRPO, 95% CI [+2.6, +4.4]", GOLD), ("+1.82", "more points from GRPO, 95% CI [+1.00, +2.66]", CORAL),
            ("20%", "of the gap to the teacher closed, for $159 of compute", CORAL)])]
  inner = ""
  for c, (name, sub, groups, teacher, stats) in enumerate(cols):
    col = box(c * 4, 4, TOP, 50, lhead(name, sub))
    col += box(c * 4, 4, TOP + 60, lad_h, ladder(groups, 533, 520, 100, teacher, True, True))
    col += "".join(box(c * 4, 4, sy0 + i * (srh + 10), srh, srow(b, cp, a)) for i, (b, cp, a) in enumerate(stats))
    inner += Bi(c, col)
  content(3, "What we got", "All three students improved", inner,
          merge_notes("""All three students improved, and every confidence interval is above zero. On BIRD, accuracy went from 30.6 to 52.2 percent, a gain of 21.6 points. On FinQA it went from 44 to 72.5 percent, a gain of 28.4 points, which is only three points short of the teacher. On MATH, Gemma went from 74.9 to 78.4 percent, a gain of 3.5 points. Look at the middle bar for MATH: it isn't trained on the teacher's data. On MATH, the best recipe trained the student on its own answers. The numbers under each chart split every gain into its steps. ~1 min.""",
                      """Let's look at BIRD. On 1,534 questions over eleven databases the student never saw, accuracy went from 30.6 to 48.2 percent after distillation, and to 52.2 after GRPO. The teacher scores 60.7. Distillation gave the biggest gain, 17.6 points, and RL added four more. The final student closed 72 percent of the gap to the teacher, and producing it cost 136 dollars of compute. ~1 min.""",
                      """FinQA was our replication. We used the same student, the same teacher, the same settings and the same six-round schedule, all fixed before the run. The only new code was the task module. On 1,147 test questions, accuracy went from 44 to 66.6 percent after distillation and to 72.5 after GRPO, 3.2 points short of the teacher. Both gains are far larger than the noise, and the final student closed 90 percent of the gap for 79 dollars. ~1 min."""),
          ["Each panel: one same-run evaluation; KD = teacher answers, Self = the student's own verified answers; dashed line = teacher; paired differences with bootstrap 95% CIs",
           "BIRD dev, 1,534 questions; FinQA test, 1,147 questions, settings and schedule unchanged from BIRD; MATH test, 5,000 problems"],
          )

  fh2 = 456
  rih = (BOT - TOP - fh2 - 16 - 2 * 10) / 3
  hb = 236
  sy1 = TOP + 58 + hb + 12
  sr2 = 74
  content(3, "What we got", "The control experiment tells us which recipe to use",
          box(0, 6, TOP, fh2, forest([
              ("BIRD", "Qwen3-1.7B, weak", 13.1, 10.7, 15.5, TEAL), ("FinQA", "Qwen3-1.7B, weak", 9.6, 6.9, 12.4, TEAL),
              ("MATH", "Gemma 4 E2B, strong", -4.3, -5.4, -3.2, GOLD)], 800, 456, -10, 20, 280)) +
          "".join(Bi(i, box(0, 6, TOP + fh2 + 16 + i * (rih + 10), rih, runin(t, b, a))) for i, (t, b, a) in enumerate([
              ("What the control is.", "We train the same SFT recipe on the same number of rows, using the student's own verified answers instead of the teacher's.", INK),
              ("The weak student learns more from the teacher.", "The teacher's answers beat the student's own by 13.1 points on BIRD and 9.6 on FinQA.", TEAL),
              ("The strong student learns more from itself.", "On MATH, the teacher's answers score 4.3 points lower than the student's own.", GOLD)])) +
          B(3, vsep() +
          head(6, 6, TOP, "A strong student lost accuracy copying the teacher") +
          box(6, 6, TOP + 58, hb, hbars([("Base", "two copies: 74.90 / 74.78", 74.9, GRAY), ("Teacher answers", "4,992 rows", 72.6, TEAL),
                                         ("Its own answers", "4,992 rows", 76.8, GOLD)], 800, hb, 100, 330, bar_h=44))) +
          B(4, "".join(box(6, 6, sy1 + i * (sr2 + 10), sr2, srow(b, c, a)) for i, (b, c, a) in enumerate([
              ("597 vs 1,027", "average tokens per answer. The distilled student copied the teacher's shorter solutions.", TEAL),
              ("−3.3 / −4.1", "points lost on the two hardest levels, where short solutions hurt most", TEAL),
              ("−10.8", "points when the teacher corrected the student's mistakes instead", INK)])) +
          box(6, 6, sy1 + 3 * (sr2 + 10) + 2, BOT - sy1 - 3 * (sr2 + 10) - 2, learn("Gemma already reasons well in its own style. Copying the teacher's shorter style cost it accuracy, while training on its own answers improved it.", CORAL, "Lesson", cls="sm"))),
          merge_notes("""If you remember one slide from this talk, make it this one. In every experiment we trained a control: the same recipe on the same number of rows, but with the student's own verified answers instead of the teacher's. The chart shows the teacher's result minus the control's.
For the weak student, the teacher's answers clearly win, by 13 points on BIRD and 10 on FinQA. For the strong student it goes the other way: the teacher's answers score 4.3 points lower than the student's own. One extra SFT run told us which recipe each student needed. ~1.5 min.""",
                      """Why did the teacher's answers hurt on MATH? Gemma already reasons well, and it writes long solutions. The teacher writes much shorter ones, about 600 tokens compared with Gemma's thousand. The distilled student copied that style and lost the most on the two hardest levels. It isn't running out of room, either: it hits the length limit less often than before.
We also tried teacher corrections, where the teacher sees the student's failed answer and writes a fix, and we train on the fixes. That was even worse, at minus 10.8 points. Training on the student's own verified answers gained about two points. ~1.5 min."""),
          ["Left: teacher-answer SFT minus self-distillation SFT, paired differences in points with bootstrap 95% CIs, same-run evaluations",
           "Right: MATH test, 5,000 problems, same-run evaluation; teacher corrections from a separate same-run evaluation"],
          also=["A strong student lost accuracy copying the teacher"])

  content(3, "What we got", "Counting cut-off answers as wrong made RL work",
          box(0, 5, TOP, 60, '<div class="lh" style="padding-left:0">Problems per round<span>with mixed outcomes</span></div>') +
          box(0, 5, TOP + 70, 400, vbars([("cut-off dropped", 600, GRAY), ("cut-off scored wrong", 1300, CORAL)], 700, 400, 1500, "~{:,.0f}")) +
          B(1, box(5, 7, TOP, 470, hbars([("Self-distilled", "starting point", 76.56, GOLD), ("+ GRPO", "cut-off answers dropped", 77.40, GRAY),
                                     ("+ GRPO", "cut-off answers scored wrong", 78.38, CORAL)], 960, 470, 100, 380, "{:.2f}%"))) +
          B(2, box(0, 12, TOP + 494, BOT - TOP - 494, learn("About 15% of the student's MATH answers hit the 2,048-token limit before giving an answer, and our original GRPO threw them away. Once we scored them as wrong, the update had twice as many problems to learn from, and GRPO added <b>1.82</b> points [+1.00, +2.66] instead of 0.84.", CORAL, "Lesson"))),
          """We had to make one more change before RL worked for the strong student. Gemma's most common failure is running out of room: about 15 percent of its answers hit the 2,048-token limit before giving a final answer. Our original GRPO threw those answers away, so the student's main failure never got a negative reward. We kept them, gave them a reward of zero, and trained only on the tokens the student actually wrote. That doubled the number of problems with mixed outcomes in each round, from about 600 to 1,300, and GRPO then added 1.82 points instead of 0.84. ~1.5 min.""",
          "MATH test, 5,000 problems, same-run evaluation; 6 GRPO rounds each; cut-off answers keep reward 0 and no end-of-turn token in the loss")

  qh = 440
  tr = (BOT - TOP - 58 - 400 - 12 - 2 * 10) / 3
  content(3, "What we got", "The trained student is both cheap and accurate",
          box(0, 6, TOP, qh, quadrant([("Teacher", 1.438, 60.7, INK, "end"), ("Untrained", 0.178, 30.6, GRAY, "start"),
                                       ("Trained student", 0.091, 52.2, CORAL, "start")], 800, 430)) +
          B(1, box(0, 6, TOP + qh + 16, BOT - TOP - qh - 16, table(["Task", "Price", "× Speed", "× Accuracy", "= Cheaper"], [
              ["BIRD", "13.6×", "1.35×", "0.86×", "<b>15.8×</b>"], ["FinQA", "13.6×", "1.35×", "0.96×", "<b>17.5×</b>"],
              ["MATH", "13.6×", "0.42×", "0.85×", "<b>4.8×</b>"]], [18, 18, 20, 23, 21], 23, accent=CORAL))) +
          B(2, vsep() +
          head(6, 6, TOP, "Why we trust these results") +
          box(6, 6, TOP + 58, 400, forest([
              ("Noise", "BIRD base vs base", 0.3, -0.8, 1.4, GRAY), ("Noise", "FinQA base vs base", -1.9, -3.3, -0.5, GRAY),
              ("Noise", "MATH base vs base", -0.1, -0.9, 0.6, GRAY), ("MATH", "self + GRPO − base", 3.5, 2.6, 4.4, CORAL),
              ("BIRD", "KD + GRPO − base", 21.6, 19.2, 24.1, CORAL), ("FinQA", "KD + GRPO − base", 28.4, 25.5, 31.5, CORAL)], 800, 440, -5, 30, 260))) +
          B(3, "".join(box(6, 6, TOP + 58 + 400 + 12 + i * (tr + 10), tr, runin(t, b)) for i, (t, b) in enumerate([
              ("Repeat runs agree.", "Three BIRD evaluations gave +17.8, +18.1 and +17.6. Four MATH evaluations stayed between +1.3 and +1.9."),
              ("No evaluation data leaked.", "A row-by-row audit found no evaluation database, report page or problem in the training data."),
              ("The results reproduce.", "A retrain produced a byte-identical model, and one script regenerates every number.")]))),
          merge_notes("""Here's the chart from the start of the talk again. On BIRD, the trained student landed where we wanted it, in the cheap and accurate corner. It's 15.8 times cheaper per correct answer than the teacher. The table covers all three tasks. Hardware price gives a factor of 13.6 everywhere, and throughput and accuracy decide the rest. FinQA reaches 17.5 times. MATH reaches only 4.8 times, because Gemma's long answers make one chip slower than the teacher's eight GPUs. Short answers save real money. ~1.5 min.""",
                      """Why should you believe these numbers? Every gain sits next to its own noise floor, which we measured in the same run with two copies of the untrained model. The SQL and finance gains are about ten times the noise. The MATH gain is small, 3.5 points, but its interval is well clear of the noise. Repeat evaluations agree within a point, the audit found no leaks, a retrain was byte-identical, and one script regenerates every number. ~1.5 min."""),
          ["Chart: BIRD. Teacher at its cheapest listed H200 rate, students at the on-demand TPU rate; cost ratio = 13.6 × throughput × accuracy",
           "Right: paired differences in points with bootstrap 95% CIs; each row from a same-run evaluation"],
          also=["Why we trust these results"])

  # ---------------- 5. Fred
  dark_divider(5, "Fred teaches<br>models to draw", "Fred took a different route to the same goal: a small model that does a specialist job well.", SKY,
               "Part 5: Fred's experiment. ~7 minutes.")

  def placeholder(label, lines):
    return (f'<div class="ph"><div class="ph-l">{esc(label)}</div>' + "".join(f"<div class='ph-i'>{esc(x)}</div>" for x in lines)
            + '<div class="ph-n">Fred\'s details to come</div></div>')

  hh = (BOT - TOP - 30) / 2
  content(4, "Fred's experiment", "The problem Fred set out to solve",
          box(0, 7, TOP, BOT - TOP, placeholder("Visual: what a drawing task looks like", ["the prompt", "the model's drawing", "what \"correct\" means"])) +
          box(7, 5, TOP, hh, card("Why drawing", "Placeholder: why this task matters and where small models fall short.", NEUTRAL)) +
          box(7, 5, TOP + hh + 30, hh, card("What success means", "Placeholder: how a drawing is judged, and the bar set in advance.", NEUTRAL)),
          """Placeholder for Fred's experiment: the problem, why drawing, and what success means.""")

  content(4, "Fred's experiment", "Fred's training journey",
          box(0, 12, TOP, 250, flow_svg([("Task", "to fill", "", DARK), ("Data", "to fill", "", DARK), ("Check", "to fill", "", DARK),
                                         ("Train", "to fill", "", DARK), ("Refine", "to fill", "", DARK), ("Serve", "to fill", "", DARK)], CW, 250)) +
          box(0, 6, TOP + 290, BOT - TOP - 290, placeholder("Decisions Fred made", ["teacher signal", "student model", "how drawings are checked"])) +
          box(6, 6, TOP + 290, BOT - TOP - 290, placeholder("What each step taught him", ["learning 1", "learning 2", "learning 3"])),
          """Placeholder: Fred's training steps in the same flow, his decisions, and what each step taught him.""")

  content(4, "Fred's experiment", "What Fred found, and how his approach compares",
          box(0, 4, TOP, 440, placeholder("Results", ["before vs after", "example drawings", "cost to serve"])) +
          box(4, 8, TOP, 440, '<div class="cmp cmp4"><div class="cmp-h"><span></span><b>SQL · finance</b><b>Maths</b><b>Drawing · Fred</b></div>'
              '<div class="cmp-r"><span>Checked by</span><b>the grader</b><b>the grader</b><i>to fill</i></div>'
              '<div class="cmp-r"><span>Training signal</span><b>teacher answers</b><b>own answers</b><i>to fill</i></div>'
              '<div class="cmp-r"><span>Student</span><b>Qwen3-1.7B</b><b>Gemma 4 E2B</b><i>to fill</i></div>'
              '<div class="cmp-r"><span>Gain</span><b>+21.6 / +28.4</b><b>+3.5</b><i>to fill</i></div>'
              '<div class="cmp-r"><span>Cheaper</span><b>15.8× / 17.5×</b><b>4.8×</b><i>to fill</i></div></div>') +
          box(0, 12, TOP + 480, BOT - TOP - 480, learn("Placeholder: the lesson both approaches share.", SKY, "What both approaches show")),
          """Placeholder: Fred's results and the routes side by side.""")

  # ---------------- closing
  h3 = (BOT - TOP - 2 * 24) / 3
  tt_rows = [["Task", "long-horizon terminal agents; Terminal-Bench, 89 hard tasks", "single-turn SQL, finance and maths answers"],
             ["Student", "Qwen3.5-9B", "Qwen3-1.7B, Gemma 4 E2B"],
             ["Hardware", "8× B300: 2 trainer, 5 rollout, 1 eval workers", "1 TPU v6e-8 host; teacher on 8× H200"],
             ["Starting point", "RL on ~1,000 cleaned tasks", "distil verified answers, then RL"],
             ["Algorithm", "DPPO, loss averaged per prompt", "GRPO, PPO clip, KL to the start"],
             ["Sampling", "asynchronous, never pauses; weights hot-swapped", "synchronous rounds: sample, then update"],
             ["Result", "+52% on Terminal-Bench after 33 h of RL", "+21.6 / +28.4 / +3.5 points; $79–$159"]]
  content(4, "Similar approaches", "Meta's TorchTitan RL work found similar lessons",
          box(0, 8, TOP, BOT - TOP, table(["", "TorchTitan RL · Meta", "TPU-Distil"], tt_rows, [19, 42, 39], 23, accent=INK)) +
          B(1, box(8, 4, TOP, h3, learn("Only 42% of Meta's tasks were clean before repair, and 40% of our teacher's BIRD answers were wrong.", TEAL, "Both cleaned the data"))) +
          B(2, box(8, 4, TOP + h3 + 24, h3, learn("A compiled kernel gave Meta a 2.5× log-prob mismatch. We check that our trainer and server agree.", CORAL, "Both checked the numerics"))) +
          B(3, box(8, 4, TOP + 2 * (h3 + 24), h3, learn("Meta penalises a missing submission, and we score cut-off answers as wrong.", GOLD, "Both reward failures"))),
          """How does this compare with other work? At the PyTorch Conference, Meta presented TorchTitan RL for long-horizon agents. They trained a Qwen3.5 9B model with RL on about a thousand cleaned terminal tasks, using eight B300s split into trainer, rollout and eval workers that run fully asynchronously. Their problem and scale are different. They go straight to RL on multi-turn agents, while we distil first and then run RL on single-turn answers.
But the lessons are similar. Both projects cleaned the data first: only 42 percent of their tasks were clean, and 40 percent of our teacher's answers were wrong. Both checked the numerics: a compiled kernel gave them a 2.5x log-prob mismatch, and we check that our trainer and server agree. And both had to reward failures carefully. They treat a dead sandbox, a missing submission and the context limit differently, and our MATH fix, counting cut-off answers as wrong, is the same idea. ~2 min.""",
          "Meta figures from the speaker's notes of \"Training long-horizon agents with TorchTitan RL\", PyTorch Conference 2026; TPU-Distil figures from the README")

  rsteps = [("Measure", "Score the teacher and a few candidate students with one prompt, and measure the noise floor.", "grade.py · gold + shuffled controls", CORAL),
            ("Generate", "Sample the teacher 4 times and the student 8 times per question, at temperature 0.7.", "generate.py", GOLD),
            ("Verify", "Keep the answers your grader accepts, up to 4 distinct per question, and audit for leaks.", "build_sft.py · leak_audit.py", TEAL),
            ("Distil", "Train LoRA on both data sets (rank 256, lr 1e-4, batch 256, 3 epochs) and keep the better one.", "train_tpu_vm.sh sft", BLUE),
            ("Reinforce", "Run GRPO with 8 samples at temperature 1.0, and count cut-off answers as wrong.", "build_grpo_rollouts.py --keep-truncated", CORAL),
            ("Serve and price", "Serve on vLLM-TPU with one copy per chip, then work out the cost per correct answer.", "serve_small.sh · bench_serving.py", SKY)]
  sh2, tk = 132, 176
  ry = TOP + sh2 + 12 + tk + 12
  choose_h = 76
  rch = BOT - ry - 58 - 12 - choose_h
  rcards = "".join(box(i * 2, 2, ry + 58, rch,
                       f'<div class="rstep sm" style="border-top:8px solid {STEP_COLOR[i]}"><div class="rstep-h"><b style="background:{STEP_COLOR[i]};color:{ON.get(STEP_COLOR[i], INK)}">{i + 1}</b>{esc(n)}</div>'
                       f'<div class="rstep-d">{esc(d)}</div><div class="rstep-s">{esc(sc)}</div></div>')
                   for i, (n, d, sc, c) in enumerate(rsteps))
  content(4, "Takeaways", "What we learned",
          box(0, 4, TOP, sh2, stat('+22 / +28 / +3.5', "points gained on SQL, finance and maths by students of about 2B parameters", CORAL, cls="sm")) +
          box(4, 4, TOP, sh2, stat("4.8–17.5×", "cheaper per correct answer than the 552B teacher", DARK, cls="sm")) +
          box(8, 4, TOP, sh2, stat("$79–$159", "of compute to produce each student on one TPU host", INK, cls="sm")) +
          B(1, box(0, 12, TOP + sh2 + 12, tk, '<div class="take sm">'
              '<div><b>If the student is weak, distil the teacher.</b> The teacher\'s verified answers gave most of the gain.</div>'
              '<div><b>If the student is strong, let it learn from itself.</b> Its own verified answers beat the teacher\'s.</div>'
              '<div><b>Then add RL.</b> Make sure the student\'s main failure gets a negative reward.</div>'
              '<div><b>Measure carefully.</b> Compare models in one run, with a control and a noise floor, and price each correct answer.</div></div>')) +
          B(2, head(0, 12, ry, "How to run this recipe on your own task", "github.com/jawadaminGOOG/tpu-distil") +
          rcards) +
          B(3, box(0, 12, BOT - choose_h, choose_h, runin("How to choose the recipe.", "Train on the teacher's answers and on the student's own answers side by side. If the teacher's answers win, use KD + GRPO. If the student's own answers win, use self-distillation + GRPO, which needs no teacher time.", TEAL))),
          merge_notes("""To wrap up: we used one pipeline on three tasks with two students. A 1.7-billion-parameter student gained 22 points on SQL and 28 on finance, and a strong Gemma student gained 3.5 points on competition maths. The students are 4.8 to 17.5 times cheaper per correct answer than the 552-billion-parameter teacher, and each one cost 79 to 159 dollars of compute to produce.
If your student is weak, distil the teacher. If it's strong, let it learn from its own answers. Then add RL, and make sure the student's main failure gets a negative reward. And measure carefully. ~1 min.""",
                      """Here's the recipe, so you can try it on your own task. Measure the gap and the noise with one prompt. Generate two training sets, one from the teacher and one from the student itself. Keep only the answers your grader accepts, and check for leaks. Distil both sets with the same LoRA settings and keep the better one. That one comparison tells you whether your student is weak or strong. Then run GRPO with the grader as the reward, and count cut-off answers as wrong. Finally, serve on single chips and work out the price of a correct answer.
The code you write is one task module. It loads the questions, builds a versioned prompt, extracts the answer and grades it, with gold and shuffled checks. We added FinQA and MATH exactly this way.
Thank you; I'm happy to take questions. ~2 min."""),
          "Code, settings and commands: github.com/jawadaminGOOG/tpu-distil (README recipe, docs/RECIPE.md)",
          also=["How to run this recipe on your own task"])

  build_appendix()


# ------------------------------------------------------------------ appendix

def build_appendix():
  H = BOT - TOP
  add(f'''<div class="bignum" style="left:{ML - 8}px;top:110px;-webkit-text-stroke:3px {SKY}">A</div>
<h1 class="div-title" style="left:{ML}px;top:450px;width:1080px">Appendix</h1>
<div class="div-sub" style="left:{ML}px;top:760px;width:980px">Backup slides for questions. Each one names the main slide it supports.</div>
<div class="circle" style="left:1290px;top:170px;width:520px;height:520px;background:{SKY}"></div>
<div class="circle" style="left:1210px;top:610px;width:200px;height:200px;background:{GOLD};opacity:.92"></div>
<div class="circle" style="left:1640px;top:700px;width:110px;height:110px;background:{CORAL}"></div>''',
      "Appendix: backup slides for questions. Each slide names the main slide it backs.", "dark")
  map_slide = len(SLIDES)
  add("", "")  # placeholder, filled after the appendix is built
  h2 = (H - 24) / 2
  h3 = (H - 2 * 24) / 3

  def grid6(items):
    return "".join(box((i % 3) * 4, 4, TOP + (i // 3) * (h2 + 24), h2, card(t, b)) for i, (t, b, _) in enumerate(items))

  # ---- the problem
  appendix(0, "A1", "whole GPU server", "The teacher ties up a whole server, even when idle",
           box(0, 7, TOP, H, table(["Teacher deployment", ""], [
               ["Endpoint", "Vertex AI dedicated endpoint, a3-ultragpu-8g (8× H200), us-south1"],
               ["Container", "Model Garden vLLM <span class=mono>pytorch-vllm-serve</span>, release 20260922"],
               ["Layout", "tensor parallel 8 with expert parallelism, fp8 KV cache"],
               ["Limits", "65,536-token context, 128 concurrent sequences"],
               ["Mode", "non-thinking (<span class=mono>thinking: false</span>)"],
               ["Cold start", "about 25 minutes from zero; a keep-warm request every 5 minutes"]], [28, 72], 24, accent=INK)) +
           box(7, 5, TOP, h3, stat("38 min", "of teacher generation the BIRD recipe needed (5.1 H200-hours)", INK)) +
           box(7, 5, TOP + h3 + 24, h3, stat("5.3 h", "that the endpoint stayed up during the BIRD work, costing 42.5 GPU-hours or $194", INK)) +
           box(7, 5, TOP + 2 * (h3 + 24), h3, learn("A dedicated endpoint keeps billing while it sits idle. A batch job sized to the actual generation would cost far less.", TEAL, "Lesson")),
           "The teacher needs a whole 8-GPU server, takes about 25 minutes to start, and bills while idle.",
           "docs/RECIPE.md (serving), docs/COMPUTE.md (endpoint uptime)")

  appendix(0, "A2", "correct answer has a price", "How we price a correct answer",
           box(0, 7, TOP, 250, '<div class="formula"><div class="fx">$ per 1,000 correct = 1,000 × $/hour ÷ (requests/s × 3,600 × accuracy)</div>'
               '<div class="fn">requests/s: best error-free rate over a concurrency ladder, on the task\'s own prompts</div></div>') +
           box(0, 7, TOP + 274, H - 274, table(["Model, BIRD", "$/hour", "req/s", "Accuracy", "$ / 1,000 correct"], [
               ["Teacher, 8× H200", "36.60", "11.7", "60.7%", "1.438"],
               ["Untrained, 1 v6e chip", "2.70", "13.8", "30.6%", "0.178"],
               ["Trained student, 1 v6e chip", "2.70", "15.8", "52.2%", "0.091"]], [36, 14, 13, 16, 21], 24, True, CORAL)) +
           box(7, 5, TOP, h2, card("The pricing favours the teacher", "We price the TPU at its on-demand rate and the teacher at the cheapest listed H200 rate (DWS flex-start), without CPU or memory. At the same purchase tier the gap would roughly double.", NEUTRAL)) +
           box(7, 5, TOP + h2 + 24, h2, card("How fast the teacher would need to be", "The student loses its cost advantage only if the teacher served 15.8× faster on BIRD, 17.5× on FinQA or 4.8× on MATH.", NEUTRAL)),
           "This is the formula behind every cost number, with the measured inputs. The pricing deliberately favours the teacher.",
           "Prices from the Cloud Billing Catalog, effective 2026-10-09")

  appendix(0, "A3", "Generic small models", "The prompt the student sees on BIRD",
           box(0, 7, TOP, H, code("SYSTEM  You write SQLite queries.\n\nUSER    1  CREATE TABLE statements\n        2  3 example rows per table,\n           each cell cut to 40 chars\n        3  BIRD external-knowledge hint\n        4  the question\n        5  reason briefly, then one\n           ```sql``` block", TEAL, "Prompt template · SHA-256 recorded per run")) +
           box(7, 5, TOP, H, table(["BIRD data", ""], [
               ["Evaluation", "dev: 1,534 questions on 11 databases, never trained on"],
               ["Training", "64 of 69 training databases, 8,432 questions"],
               ["Removed", "<span class=mono>works_cycles</span>: 64,016-character schema"],
               ["Hold-out", "4 training databases, 522 questions, for screens"],
               ["Grader", "standard-library sqlite3, read-only copy, 30 s timeout"]], [32, 68], 23, accent=CORAL)),
           "The prompt is fixed and versioned. It holds the schema, example rows, the hint, the question and a format instruction.",
           "docs/RECIPE.md §1; system message paraphrased")

  # ---- the experiments
  appendix(1, "A4", "automatic grader", "How the FinQA task is set up",
           box(0, 6, TOP, H, code("SYSTEM  You are an expert financial\n        analyst.\n\nUSER    report text before the table\n        the table, pipe-separated rows\n        report text after the table\n        the question\n        reason briefly, end with\n        Answer: <number>\n        (yes/no only for yes/no questions)", TEAL, "Prompt finqa-2 · SHA-256 recorded")) +
           box(6, 6, TOP, 420, table(["FinQA", ""], [
               ["Splits", "train 6,251 · dev 883 · test 1,147 (evaluation)"],
               ["Leak filter", "142 training questions removed (exact or 13-word overlap); no shared report pages"],
               ["Grader", "within 1%, or 0.005 absolute below 0.5; ×100 and ÷100 forms"],
               ["Controls", "gold 100% · shuffled 0.6%"]], [26, 74], 22, accent=TEAL)) +
           box(6, 6, TOP + 444, H - 444, learn("Our first prompt offered a yes/no answer on every question, and the base student answered yes or no on 363 numeric ones. We replaced the prompt, using dev, before generating any teacher answers.", CORAL, "Lesson")),
           "This shows what the student sees on FinQA and how its answer is graded.", "docs/RECIPE.md §8; README")

  appendix(1, "A5", "automatic grader", "How the MATH task is set up",
           box(0, 6, TOP, H, code("SYSTEM  You are an expert\n        mathematician.\n\nUSER    the problem\n        reason briefly, end with\n        \\boxed{<answer>}\n\nGRADE   last \\boxed{...}, balanced braces\n        reference strip_string normalisation:\n        spacing, \\left/\\right, dfrac, units,\n        degrees, percent, 0.5 = \\frac{1}{2}", BLUE, "Prompt math-1 · SHA-256 recorded")) +
           box(6, 6, TOP, 420, table(["MATH", ""], [
               ["Source", "EleutherAI/hendrycks_math, MIT licence; 7 subjects, 5 levels"],
               ["Splits", "test 5,000 (evaluation) · dev 500 held out by seed, by subject"],
               ["Leak filter", "13 identical and 725 13-word overlaps removed: 6,262 remain"],
               ["Controls", "gold 100% · shuffled within subject 0.0%"]], [26, 74], 22, accent=BLUE)) +
           box(6, 6, TOP + 444, H - 444, learn("Exact matching counts equal answers written differently, like 2√2 and √8, as wrong. This affects every model equally.", GOLD, "Caveat")),
           "We added the MATH task module the same way as FinQA.", "docs/RECIPE.md §9; README")

  appendix(1, "A6", "starts weak", "A closer look at the two students",
           box(0, 7, TOP, H, table(["", "Qwen3-1.7B", "Gemma 4 E2B-it"], [
               ["Used on", "BIRD, FinQA", "MATH"],
               ["Size", "1.41B decoder matmul + 0.31B tied head", "35 layers, about 2B effective"],
               ["LoRA", "rank 256 on 7 projections, 196 modules, 0.28B trainable", "rank 256, 190 modules incl. fused k/v"],
               ["Base accuracy", "30.6% BIRD · 44.0% FinQA", "74.9% MATH"],
               ["Teacher", "60.7% BIRD · 75.7% FinQA", "92.5% MATH"],
               ["Licence", "open weights", "Apache-2.0"]], [24, 40, 36], 23, accent=GOLD)) +
           box(7, 5, TOP, h2, card("Qwen3-4B", "It already scores 61.7% on BIRD without training. We didn't measure its serving cost.", GRAY)) +
           box(7, 5, TOP + h2 + 24, h2, learn("Picking the right student can matter as much as distilling it.", GOLD, "Lesson")),
           "Both student families went through the same trainer.", "README, docs/COMPUTE.md, docs/RECIPE.md §9")

  appendix(1, "A7", "Six decisions", "We fixed the evaluation design before seeing results",
           grid6([("Targets set in advance", "We wrote down the pass conditions before each stage.", CORAL),
                  ("Test sets stayed untouched", "No evaluation set chose a checkpoint or a hyperparameter, and we fixed the final checkpoint in advance.", GOLD),
                  ("One run per comparison", "We serve and grade every compared model together, with greedy decoding and a 2,048-token limit.", TEAL),
                  ("Two copies of the baseline", "Every run includes two copies of the untrained model, which give the noise floor.", BLUE),
                  ("Paired statistics", "We compare models question by question and report bootstrap 95% CIs from 10,000 resamples.", SKY),
                  ("Early stopping", "BIRD used only its 522 held-out training-database questions for early stopping.", CORAL)]),
           "This is how we avoided tuning on the test set.", "docs/RECIPE.md §7, README limits")

  appendix(1, "A8", "Six decisions", "Answer length is a serving cost",
           box(0, 7, TOP, H, hbars([("FinQA", "Qwen3-1.7B · ~134-token answers", 17.5, CORAL), ("BIRD", "Qwen3-1.7B · short SQL answers", 15.8, CORAL),
                                    ("MATH", "Gemma 4 E2B · ~980-token answers", 4.8, CORAL)], 1040, 640, 20, 380, "{:.1f}×", 60)) +
           box(7, 5, TOP, h2, card("Why we turned thinking off", "Teacher and student both answer briefly. The student would copy long reasoning traces, and the teacher samples 4× faster without them.", NEUTRAL)) +
           box(7, 5, TOP + h2 + 24, h2, learn("On MATH the student's answers are 2.3× longer than the teacher's, so one chip finishes fewer requests than eight GPUs, and throughput drops to 0.42×.", CORAL, "Lesson")),
           "The cost advantage per correct answer depends on answer length.", "Bars: teacher ÷ student cost per correct answer (README)")

  # ---- the journey
  appendix(2, "A9", "Check the measurements", "Every grader passed two checks",
           box(0, 12, TOP, 430, table(["Task", "Gold answers", "Shuffled answers", "How an answer is graded"], [
               ["BIRD", "100% of 1,530 executable", "0.2%", "run the last SQL block on a read-only copy; compare row sets"],
               ["FinQA", "100%", "0.6%", "number after the last <span class=mono>Answer:</span>, within 1%; ×100 and ÷100 forms accepted"],
               ["MATH", "100%", "0.0%", "last <span class=mono>\\boxed{}</span> after the reference normalisation"]], [12, 22, 18, 48], 25)) +
           box(0, 12, TOP + 460, H - 460, learn("Before any accuracy number means anything, the grader has to score gold answers at 100% and shuffled answers near 0%. FinQA's grader is lenient, so its scores aren't comparable to published FinQA results.", CORAL, "Lesson")),
           "Every grader passes two checks before we use it.", "README, docs/RECIPE.md")

  appendix(2, "A10", "Choose who writes", "How much training data each task produced",
           box(0, 12, TOP, 430, table(["Set", "Questions", "Samples", "Correct", "Rows kept", "Time"], [
               ["BIRD, teacher ×4", "8,432", "33,728", "20,133 (60%)", "10,112", "38 min"],
               ["FinQA, teacher ×4", "6,109", "24,436", "18,487 (76%)", "6,016", "26 min"],
               ["MATH, teacher ×4", "6,262", "25,048", "93.5%", "4,992 of 6,021", "—"],
               ["MATH, student ×8", "6,262", "50,096", "72.6%", "4,992 of 5,237", "no teacher"]], [24, 13, 13, 18, 18, 14], 24, accent=GOLD)) +
           box(0, 6, TOP + 460, H - 460, card("How we sampled", "We sampled at temperature 0.7 (top-p 0.95 on FinQA and MATH), in non-thinking mode, on training questions only. Each control uses the same number of the student's own answers.", NEUTRAL)) +
           box(6, 6, TOP + 460, H - 460, learn("On MATH, all correct answers to a problem reduce to the same normalised answer, so the 4-distinct limit keeps one row per problem.", TEAL, "Note")),
           "This shows how much training data each task produced and how much of it passed the grader.", "docs/COMPUTE.md, docs/RECIPE.md §4, §8, §9")

  appendix(2, "A11", "Keep only the answers", "The leak audit found nothing",
           box(0, 6, TOP, H, table(["The audit rejects", ""], [
               ["Evaluation groups", "BIRD databases, FinQA report pages, MATH problem fingerprints"],
               ["Question overlap", "exact match or a shared 13-word sequence with an evaluation question"],
               ["Non-training ids", "any id outside the training split"],
               ["Stale answers", "any kept answer that no longer verifies on re-grading"],
               ["In the trainer", "ids from <span class=mono>--eval_tasks</span> are refused again"]], [34, 66], 23, accent=TEAL)) +
           box(6, 6, TOP, h2, stat("0 leaks", "found on BIRD, FinQA and MATH, across rows, groups, overlaps and re-grades", DARK)) +
           box(6, 6, TOP + h2 + 24, h2, card("Some questions were removed up front", "We removed 142 FinQA training questions and 738 MATH problems before generation. Much of MATH's wording is shared boilerplate, so its filter is strict.", NEUTRAL)),
           "Three layers keep evaluation data out of training: the prompt builder, the record builder with its audit, and the trainer.", "docs/ARCHITECTURE.md, docs/RECIPE.md")

  mask = svg(1680, 200, "".join([
      R(0, 40, 300, 90, GRAY, 12), T(150, 95, "system", 26, 700, INK, "middle"),
      R(310, 40, 720, 90, GRAY, 12), T(670, 95, "user: the task's prompt", 26, 700, INK, "middle"),
      R(1040, 40, 470, 90, TEAL, 12), T(1275, 95, "response", 26, 800, "#ffffff", "middle"),
      R(1520, 40, 160, 90, TEAL, 12), T(1600, 95, "end-of-turn", 22, 800, "#ffffff", "middle"),
      T(670, 175, "loss weight 0", 24, 700, INK, "middle"), T(1360, 175, "loss weight 1", 24, 700, DARK, "middle")]), bare=True)
  appendix(2, "A12", "Keep only the answers", "From graded answers to training records",
           box(0, 12, TOP, 200, mask) +
           box(0, 7, TOP + 230, H - 230, code('{"id": "...#s0",\n "db_id" | "report" | "problem_hash",\n "messages": [system, user],\n "response": "..."}\n\nGRPO adds: "group_id", "reward",\n           "truncated" (cut-off answers)', BLUE, "One JSONL record")) +
           box(7, 5, TOP + 230, H - 230, table(["Builder rules", ""], [
               ["Distinct", "at most 4 distinct answers per question"],
               ["Exact N", "drawn to a fixed size with a fixed seed"],
               ["Template check", "records the chat template alters are dropped and counted"],
               ["Cut-off answers", "no end-of-turn token in the loss"]], [38, 62], 22, accent=BLUE)),
           "The loss covers only what the model should produce, and every record renders to exactly the prompt that vLLM serves.", "docs/ARCHITECTURE.md (record format)")

  appendix(2, "A13", "Distil the student", "Supervised fine-tuning settings",
           box(0, 8, TOP, H, table(["Setting", "Value"], [
               ["Student weights", "bf16, frozen"],
               ["LoRA", "rank 256, alpha 512, attention and MLP projections, every layer; float32 adapters"],
               ["Batch", "256 sequences: micro-batch 8 × gradient accumulation 32"],
               ["Optimiser", "AdamW, lr 1e-4, 5% warm-up, cosine to 0, clip 1.0"],
               ["Length", "≤ 8,192 tokens; buckets 2,048 / 4,096 / 8,192"],
               ["Schedule", "3 epochs: 117 steps (BIRD), 69 (FinQA), 57 (MATH)"],
               ["Mesh", "FSDP 8 × TP 1; rematerialisation; splash attention"]], [28, 72], 23, accent=BLUE)) +
           box(8, 4, TOP, h2, stat("10–27 min", "SFT wall time on the v6e-8: MATH 10, FinQA 13, BIRD 27", DARK)) +
           box(8, 4, TOP + h2 + 24, h2, learn("Every size is a multiple of 256 to match the v6e's 256 × 256 matrix unit.", TEAL, "Rule")),
           "We used one SFT configuration for every task and both students.", "docs/RECIPE.md §5, docs/COMPUTE.md")

  appendix(2, "A14", "Distil the student", "The trainer and the server compute the same model",
           box(0, 4, TOP, h2, stat("0.998", "Qwen3 top-1 agreement, trainer vs vLLM, 16 prompts (limit 0.99); KL 1e-4", DARK)) +
           box(4, 4, TOP, h2, stat("0.994", "Gemma 4 top-1 agreement on vLLM's own continuations; KL 5.8e-4", DARK)) +
           box(8, 4, TOP, h2, stat("1 : 1", "retrain: identical loss at all 117 steps, byte-identical checkpoint", DARK)) +
           box(0, 6, TOP + h2 + 24, h2, card("The prompts match token for token", "A test on the real tokenizer checks that the training context equals the prompt vLLM serves; Gemma also passes vLLM's /tokenize check.", NEUTRAL)) +
           box(6, 6, TOP + h2 + 24, h2, card("Every model uses the same flags", "Merged checkpoints serve with the base model's flags, so compared models differ only in weights.", NEUTRAL)),
           "Before trusting any result, we checked that the training and serving models agree, and we ran an exact retrain.", "trainer/parity_check.py; docs/RECIPE.md §5, §9")

  appendix(2, "A15", "Distil the student", "What it took to add Gemma 4",
           box(0, 7, TOP, H, table(["Problem", "Fix"], [
               ["vLLM-TPU warm-up fails on the default PyTorch model path", "<span class=mono>MODEL_IMPL_TYPE=flax_nnx</span>: the JAX Gemma 4 model"],
               ["System message rendered as a Python list in 8 of 16 prompts", "<span class=mono>--chat-template-content-format string</span>; earlier outputs rerun"],
               ["vLLM refuses prompt log-probs for multimodal models", "parity on vLLM's own 64-token continuation"],
               ["qwix traces LoRA on a 2-token input splash attention refuses", "trace with flash attention off, then restore it"]], [52, 48], 22, accent=GOLD)) +
           box(7, 5, TOP, h2, card("Trainer changes", "Tunix's Gemma 4 model handles per-layer embeddings and KV sharing. LoRA covers the fused k/v in the 15 layers that compute K/V, and the logit soft-cap runs in float32.", NEUTRAL)) +
           box(7, 5, TOP + h2 + 24, h2, learn("Adding a new model family took one loader, one exporter and four serving fixes. The first start compiles for about 17 minutes per model.", TEAL, "Lesson")),
           "This is what it took to add a second model family.", "docs/RECIPE.md §9, problems and fixes")

  appendix(2, "A16", "always right", "What happens in one GRPO round",
           box(0, 12, TOP, 250, flow_svg([("Serve", "7–8 replicas of the", "current student", TEAL), ("Sample", "8 answers per question", "T 1.0, top-p 1.0", GOLD),
                                          ("Grade", "reward 1 if verified,", "else 0", CORAL), ("Select", "≤ 2,048 questions", "with mixed rewards", BLUE),
                                          ("Update", "64 sequences per step", "one pass", SKY)], CW, 250)) +
           box(0, 12, TOP + 280, H - 280, table(["Setting", "Value", "Setting", "Value"], [
               ["Advantage", "group-relative ÷ group std", "Optimiser", "AdamW; lr 2e-6 rounds 1–2, 5e-6 rounds 3–6"],
               ["Policy loss", "PPO clip 0.2, dual clip 3", "Length", "≤ 8,192 tokens"],
               ["KL", "k3 vs the starting checkpoint, β 0.02", "Guard", "stop if a step's KL > 0.2"]], [15, 30, 13, 42], 23, accent=CORAL)),
           "Each round serves the current student, samples, grades, keeps the questions with mixed outcomes, updates, merges and repeats.", "docs/RECIPE.md §6")

  appendix(2, "A17", "always right", "RL stayed stable",
           box(0, 7, TOP, 60, '<div class="lh" style="padding-left:0">FinQA steps per round<span>fewer questions have mixed outcomes each round</span></div>') +
           box(0, 7, TOP + 70, H - 70, vbars([("1", 255, CORAL), ("2", 255, CORAL), ("3", 254, CORAL), ("4", 236, CORAL), ("5", 210, CORAL), ("6", 193, CORAL)], 1040, 640, 300)) +
           box(7, 5, TOP, h3, stat("≤ 0.05", "the highest KL in any step, round or task (BIRD 0.033, FinQA 0.049, MATH 0.017)", DARK)) +
           box(7, 5, TOP + h3 + 24, h3, stat("0", "times the KL guard (0.2) had to stop a run", DARK)) +
           box(7, 5, TOP + 2 * (h3 + 24), h3, learn("At lr 2e-6, two rounds gave 1.7 to 2.1 points on BIRD. Raising it to 5e-6 for four more rounds reached 4.0.", CORAL, "Lesson")),
           "RL stayed close to the starting model. As the student improved, fewer questions had mixed outcomes.", "docs/RECIPE.md §6, §8, §9")

  appendix(2, "A18", "always right", "What RL doesn't do",
           box(0, 4, TOP, h2, stat("+0.4", "BIRD RL gain on 522 held-out training-database questions, CI [−2.3, +2.9], which is too small to detect", MUTED)) +
           box(4, 4, TOP, h2, stat("73.1 → 74.3", "BIRD best-of-8 on the training pool across 5 rounds, almost unchanged", INK)) +
           box(8, 4, TOP, h2, stat("133 → 159", "mean FinQA answer tokens under RL (+19%)", CORAL)) +
           box(0, 12, TOP + h2 + 24, h2, learn("RL sharpened answers the student could already reach instead of teaching new ones. GRPO added 1.82 points to the MATH KD student but only brought it back level with base, so RL can't make up for the wrong SFT data.", CORAL, "Lesson")),
           "These are the limits of the RL stage.", "README; distil/passk.py")

  appendix(2, "A19", "runs on a single chip", "How we measured serving throughput",
           box(0, 7, TOP, H, table(["", "Student", "Teacher"], [
               ["Software", "vLLM 0.26.1rc1 TPU build", "Model Garden vLLM"],
               ["Layout", "1 replica per v6e chip, TP 1", "TP 8 + expert parallel"],
               ["KV cache", "bf16", "fp8"],
               ["BIRD best", "15.8 req/s", "11.7 req/s, flat 128 → 512"],
               ["FinQA best", "24.1 req/s, still rising at 256", "17.9 req/s at 128"],
               ["MATH best", "2.79 req/s at 128", "6.64 req/s at 128"]], [26, 37, 37], 23, accent=SKY)) +
           box(7, 5, TOP, h2, card("Method", "We raised concurrency until errors appeared and priced the best error-free rate. Each model ran in its own session on the same hardware and build.", NEUTRAL)) +
           box(7, 5, TOP + h2 + 24, h2, learn("The BIRD and FinQA benchmarks cap answers at 1,024 tokens. MATH uses 2,048, because a 1,024 cap would cut off 36% of the student's answers.", CORAL, "Note")),
           "This is how we measured throughput for the cost numbers.", "README serving tables, docs/RECIPE.md §2")

  appendix(2, "A20", "runs on a single chip", "What broke, and how we fixed it",
           box(0, 12, TOP, H, table(["Problem", "Fix"], [
               ["Out of memory with two 8,192-token sequences per chip", "one sequence per chip; batch of 256 kept"],
               ["A prompt log-prob request (≈ 4.6 GB extra) stopped a serving replica", "parity runs on a separate replica with less cache"],
               ["A chip's runtime stopped: SLICE_FAILURE_WORKER_UNAVAILABLE", "a watchdog restarts replicas; the client retries"],
               ["Two copies of the same weights gave different text on 45% of FinQA questions", "same-run comparisons; two base copies per run"],
               ["Teacher takes about 25 minutes to start from zero", "keep-warm loop during teacher stages"],
               ["The TPU VM was unavailable for 7 hours during MATH rollouts", "disclosed; a driver should stop when any rollout fails"]], [58, 42], 23, accent=CORAL)),
           "These are the operational problems we hit and how the pipeline handles them.", "docs/RECIPE.md, problems and fixes")

  # ---- results
  appendix(3, "A21", "All three students improved", "All BIRD comparisons and training-data options",
           box(0, 7, TOP, H, table(["BIRD dev, same run", "Δ points [95% CI]", "Better / worse"], [
               ["KD + GRPO − base", "<b>+21.6</b> [+19.2, +24.1]", "387 / 55"],
               ["KD − base", "<b>+17.6</b> [+15.2, +20.1]", "334 / 64"],
               ["KD + GRPO − KD", "<b>+4.0</b> [+2.2, +5.9]", "136 / 74"],
               ["Base − base copy", "+0.3 [−0.8, +1.4]", "39 / 34"],
               ["Teacher − KD + GRPO", "+8.5 [+6.0, +10.9]", "261 / 131"]], [44, 34, 22], 23, accent=BLUE)) +
           box(7, 5, TOP, H, table(["Training data (separate run)", "BIRD"], [
               ["KD: 10,112 teacher answers", "<b>48.7%</b>"],
               ["CORR: 5,883 teacher corrections", "37.1%"],
               ["Self: 10,112 own answers", "35.6%"],
               ["Base", "30.6%"]], [70, 30], 23, accent=TEAL)),
           "Every BIRD comparison, with per-question win and loss counts, and the training-data options. CORR beat self-distillation by only 1.5 points [−1.0, +3.9].", "README")

  appendix(3, "A22", "All three students improved", "All FinQA comparisons and the control",
           box(0, 7, TOP, H, table(["FinQA test, same run", "Δ points [95% CI]", "Better / worse"], [
               ["KD + GRPO − base", "<b>+28.4</b> [+25.5, +31.5]", "367 / 41"],
               ["KD − base", "<b>+22.6</b> [+19.4, +25.7]", "321 / 62"],
               ["KD + GRPO − KD", "<b>+5.8</b> [+3.9, +7.8]", "100 / 33"],
               ["Base − base copy", "−1.9 [−3.3, −0.5]", "23 / 45"],
               ["Teacher − KD + GRPO", "+3.2 [+1.0, +5.4]", "105 / 68"]], [44, 34, 22], 23, accent=BLUE)) +
           box(7, 5, TOP, h2, card("The control (earlier run)", "Self-distillation scored 57.0% and KD 66.6%. KD beat self by 9.6 points [+6.9, +12.4], and self beat base by 14.6 [+11.8, +17.4].", GOLD)) +
           box(7, 5, TOP + h2 + 24, h2, learn("The teacher's share is smaller than on BIRD because part of FinQA's self-distillation gain comes from learning the answer format. The base student gives no number at all on more than 100 test questions.", TEAL, "Why the control gains more")),
           "Every FinQA comparison, and why the control gains more here than on BIRD.", "README")

  appendix(3, "A23", "strong student lost", "Every MATH result",
           box(0, 12, TOP, H, table(["Same-run evaluation, MATH test", "Accuracy", "Δ [95% CI]", "Cut-off answers"], [
               ["Base, two copies", "74.90 / 74.78", "−0.12 [−0.88, +0.62]", "—"],
               ["KD: 4,992 teacher answers", "72.56", "−2.34 [−3.44, −1.22] vs base", "577 (base 800)"],
               ["Self-distillation: 4,992 own answers", "76.84", "+1.94 [+1.08, +2.84] vs base", "—"],
               ["Self + GRPO, cut-off scored wrong", "78.38", "+1.82 [+1.00, +2.66] vs self", "674"],
               ["Self + GRPO, cut-off dropped", "77.40", "+0.84 [+0.02, +1.68] vs self", "756"],
               ["KD + GRPO", "≈ base", "+1.82 [+0.80, +2.86] vs KD", "—"],
               ["Self + GRPO − KD + GRPO (separate run)", "", "+2.82 [+1.78, +3.88]", ""]], [38, 14, 32, 16], 22, accent=GOLD)),
           "Every MATH result. The KD route doesn't catch up, even after GRPO.", "README, MATH section")

  appendix(3, "A24", "strong student lost", "Teacher corrections hurt on MATH",
           box(0, 7, TOP, H, table(["CORR on MATH", ""], [
               ["Input to the teacher", "the student's first failed answer and one sentence about it; never the gold answer"],
               ["Failures", "2,793 problems: 1,620 cut off, 653 wrong answer, 520 no boxed answer"],
               ["Rewrites", "4 per problem, 74.9% correct, 2,362 rows"],
               ["Result", "<b>64.26%</b>: −10.76 [−11.96, −9.54] vs base"],
               ["Length", "559 tokens on average vs self-distillation's 1,013"]], [30, 70], 23, accent=INK)) +
           box(7, 5, TOP, h2, stat("8%", "of corrected answers carry correction language: \"the student's error was…\"", CORAL)) +
           box(7, 5, TOP + h2 + 24, h2, learn("The corrections are short, written in the teacher's style, and come only from hard problems. They pull a strong student furthest from its own reasoning.", GOLD, "Lesson")),
           "Teacher corrections helped a weak student a little on BIRD and hurt a strong student badly on MATH.", "README; docs/RECIPE.md §9")

  appendix(3, "A25", "Counting cut-off answers", "How we score cut-off answers",
           box(0, 6, TOP, H, code("rollout hits 2,048 tokens\n  no end-of-turn token emitted\n\nbefore: dropped from its group\n  -> the main failure never gets\n     a negative reward\n\nnow:    kept, reward 0,\n        \"truncated\": true\n  -> data.py gives the trailing\n     end-of-turn token weight 0", CORAL, "build_grpo_rollouts.py --keep-truncated")) +
           box(6, 6, TOP, h3, stat("~600 → ~1,300", "MATH problems per round with mixed outcomes", CORAL)) +
           box(6, 6, TOP + h3 + 24, h3, stat("+0.98", "scored wrong vs dropped, CI [+0.16, +1.78]", CORAL)) +
           box(6, 6, TOP + 2 * (h3 + 24), h3, learn("If we trained on a cut-off answer as written, the student would learn to stop where it never stopped. Leaving out the end-of-turn token lowers the probability of exactly the tokens it wrote.", TEAL, "Why we drop the end-of-turn token")),
           "This is how the cut-off fix works.", "docs/RECIPE.md §9 (truncated rollouts)")

  appendix(3, "A26", "both cheap and accurate", "Cost per correct answer on each task",
           box(0, 12, TOP, 380, table(["Task", "Hardware price", "× throughput", "× accuracy", "= cheaper", "Student vs teacher $ / 1,000 correct"], [
               ["BIRD", "13.6×", "1.35× (15.8 vs 11.7 req/s)", "0.86×", "<b>15.8×</b>", "0.091 vs 1.438"],
               ["FinQA", "13.6×", "1.35× (24.1 vs 17.9 req/s)", "0.96×", "<b>17.5×</b>", "0.043 vs 0.750"],
               ["MATH", "13.6×", "0.42× (2.79 vs 6.64 req/s)", "0.85×", "<b>4.8×</b>", "0.343 vs 1.656"]], [12, 15, 26, 12, 12, 23], 23, accent=BLUE)) +
           box(0, 12, TOP + 410, H - 410, learn("Hardware price gives the same 13.6× on every task, and throughput decides the rest. Short answers keep throughput above 1×, while long MATH answers cut it to 0.42×.", CORAL, "Lesson")),
           "This breaks down the cost advantage on all three tasks.", "README serving tables")

  appendix(3, "A27", "both cheap and accurate", "How much compute each task used",
           box(0, 12, TOP, 330, stacked([("BIRD", {"gen": 0, "sft": 3.6, "grpo": 25.8, "roll": 12.5}), ("FinQA", {"sft": 1.7, "grpo": 16.3, "roll": 5.1}),
                                         ("MATH", {"gen": 6.2, "sft": 1.3, "grpo": 17.4, "roll": 34.0})],
                                        [("gen", "self-samples", GOLD), ("sft", "SFT", TEAL), ("grpo", "GRPO updates", CORAL), ("roll", "GRPO rollouts", SKY)], CW, 330, 140)) +
           box(0, 4, TOP + 360, H - 360, stat("$136", "BIRD: 41.9 chip-hours + 5.1 teacher H200-hours", INK)) +
           box(4, 4, TOP + 360, H - 360, stat("$79", "FinQA: 23.1 chip-hours + 3.5 teacher H200-hours", INK)) +
           box(8, 4, TOP + 360, H - 360, stat("$159", "MATH: 58.9 chip-hours, no teacher time", INK)),
           "These are TPU v6e chip-hours per stage. SFT is cheap, RL takes most of the compute, and MATH rollouts dominate because the answers are long. Utilisation was 3-5%, so these are upper bounds.",
           "docs/COMPUTE.md, generated by distil/compute_report.py; bars in TPU v6e chip-hours")

  appendix(3, "A28", "Why we trust", "What these results don't show",
           grid6([("One training seed", "We trained each model once, so the noise floor measures serving noise, not training-seed variance.", CORAL),
                  ("Student and task changed together", "The student changed along with the task, so \"already strong\" is our best explanation, not a proven cause.", GOLD),
                  ("Public benchmarks", "The teacher and students may have seen BIRD, FinQA or MATH during pre-training. This applies to every model.", TEAL),
                  ("The FinQA grader is lenient", "It accepts a 1% tolerance and percentage forms, so our scores aren't comparable to published FinQA results.", BLUE),
                  ("We missed the MATH targets", "We required KD to gain 5 points (it lost 2.3) and each GRPO stage to gain 2 (the best gained 1.82). A 7-hour outage cut one round to 3,367 of 6,262 problems.", SKY),
                  ("The cost gap depends on deployment", "A faster teacher deployment would shrink the cost advantage (see A2).", CORAL)]),
           "These are the caveats we state up front.", "README, what this does not show")

  # ---- stack, recipe, field
  appendix(4, "A29", "open-source libraries", "Versions and machines",
           box(0, 6, TOP, H, table(["Library", "Version", "Use"], [
               ["vLLM (TPU build)", "0.26.1rc1", "student serving"],
               ["JAX · libtpu", "0.11.0 · 0.0.44", "compile, TPU runtime"],
               ["google-tunix", "0.1.8", "Qwen3 / Gemma 4 models, trainer"],
               ["qwix", "0.1.8", "LoRA adapters"],
               ["Flax NNX · Optax · Orbax", "0.12.8 · 0.2.6 · 0.12.2", "modules, AdamW, checkpoints"],
               ["transformers · safetensors", "5.12.1 · 0.7.0", "tokenizer, template, files"]], [44, 26, 30], 22, accent=SKY)) +
           box(6, 6, TOP, H, table(["Machine", "Role"], [
               ["TPU v6e-8 VM, us-east5-a", "serves, trains, merges, benchmarks the student; never serving and training at once"],
               ["CPU VM n4-standard-64", "databases, prompts, grading, data builds, long-run drivers"],
               ["Vertex AI endpoint, 8× H200, us-south1", "the teacher"],
               ["Cloud Storage bucket", "checkpoints, data, evidence"]], [42, 58], 22, accent=INK)),
           "These are the exact versions and what each machine does.", "docs/RECIPE.md, docs/ARCHITECTURE.md")

  appendix(4, "A30", "open-source libraries", "How the code is organised",
           box(0, 12, TOP, 250, flow_svg([("prompts.py", "versioned template", "per task", GOLD), ("generate.py", "teacher or student", "resumes by id", INK),
                                          ("grade.py", "task grader", "gold + shuffled", CORAL), ("build_*.py", "SFT or GRPO records", "+ leak_audit.py", TEAL),
                                          ("trainer/", "SFT / GRPO / merge", "Tunix on v6e-8", BLUE), ("report.py", "paired CIs, cost", "from raw files", SKY)], CW, 250)) +
           box(0, 6, TOP + 280, H - 280, card("distil/", "Standard-library scripts. Every task-specific step goes through <span class=mono>distil/tasks/</span> (bird, finqa, math), selected with <span class=mono>--task</span>.", NEUTRAL)) +
           box(6, 6, TOP + 280, H - 280, card("trainer/", "<span class=mono>train.py</span> picks Qwen3 or Gemma 4 from config.json, <span class=mono>data.py</span> masks the loss, and <span class=mono>parity_check.py</span> and <span class=mono>kl_guard.sh</span> check the results.", NEUTRAL)),
           "The two halves share data only through JSONL files and Hugging Face checkpoints.", "docs/ARCHITECTURE.md")

  appendix(4, "A31", "How to run this recipe", "Adding your own task or student",
           box(0, 6, TOP, H, code("# distil/tasks/<your_task>.py\nload_items(split)      -> items\nPROMPT_VERSION, render(item)\nextract(response)      -> answer\ngrade(answer, item)    -> bool\n\n# controls before use\ngold answers   ~ 100%\nshuffled       ~ 0%\n\n# then: add the name to NAMES", TEAL, "A new task is one module")) +
           box(6, 6, TOP, h2, card("Adding a new student", "Add a config builder, loader, LoRA targets and delta exporter in <span class=mono>student_models.py</span>, and its end-of-turn marker in <span class=mono>data.py</span>.", NEUTRAL)) +
           box(6, 6, TOP + h2 + 24, h2, learn("Check logit parity with parity_check.py, and check the served chat template with vLLM's /tokenize. We added FinQA, MATH and Gemma 4 this way.", CORAL, "Before you train")),
           "This is what you write to bring the recipe to a new task or model.", "docs/ARCHITECTURE.md, extension points; interface names abbreviated")

  appendix(4, "A32", "TorchTitan", "A closer look at TorchTitan RL",
           grid6([("Sampling never pauses", "Rollouts run continuously, new weights are swapped in while they run, and the buffer takes any 32 finished groups in any order.", BLUE),
                  ("The KV cache survives updates", "They keep it across weight updates with no accuracy loss. KV headroom limits how many rollouts run in parallel.", TEAL),
                  ("The agent runs outside the sandbox", "The policy controls a sandbox but doesn't run inside it.", INK),
                  ("Each failure gets its own reward", "A dead sandbox gets no penalty, a missing submission gets a penalty, and hitting the context limit gets a nudge.", CORAL),
                  ("They cleaned the tasks", "Only 42% of TMAX tasks and 61% of CalibForge tasks were clean before repair.", GOLD),
                  ("DPPO with per-prompt averaging", "They use DPPO instead of GRPO or PPO, and average the loss per prompt instead of per token.", SKY)]),
           "These are the design choices Meta described for TorchTitan RL on Terminal-Bench.",
           "Speaker's notes of \"Training long-horizon agents with TorchTitan RL\", PyTorch Conference 2026")

  appendix(4, "A33", "TorchTitan", "What we'd borrow from TorchTitan RL",
           box(0, 12, TOP, H, table(["Idea from TorchTitan RL", "Where it fits in TPU-Distil", "Why"], [
               ["Asynchronous rollouts", "overlap sampling and updates across the 8 chips", "rollouts were 30% of BIRD and 58% of MATH chip-hours, with the trainer idle"],
               ["Online task generation (+8 points vs a static pool)", "generate new questions per round from training data", "later FinQA rounds ran out of mixed-outcome questions"],
               ["Failure-type rewards", "extend the MATH cut-off rule to other failure types", "each failure mode needs its own signal"],
               ["Bitwise trainer–sampler parity", "tighten parity from 0.998 top-1 toward exact", "removes a source of off-policy drift"],
               ["Per-prompt loss averaging", "an option in the GRPO loss", "long answers stop dominating the update"]], [30, 36, 34], 23, accent=SKY)),
           "The comparison suggests these next steps.", "These are ideas; we haven't tested any of them in TPU-Distil")

  # ---- appendix map, filled in place
  ent = "".join(f'<div><b style="background:{c};color:{ON.get(c, INK)}">{esc(code_)}</b>{esc(t)}<i>→ {n}</i></div>' for code_, _, n, t, c in APPENDIX)
  rows = (len(APPENDIX) + 2) // 3
  body = (f'''<div class="kicker" style="left:{ML}px;top:70px"><span class="dot" style="background:{SKY}"></span>Appendix map</div>
<h2 class="title" style="left:{ML}px;top:104px;width:{CW}px">{len(APPENDIX)} backup slides and the main slides they support</h2>''' +
          box(0, 12, TOP - 20, H + 40, f'<div class="amap" style="grid-auto-flow:column;grid-template-rows:repeat({rows},auto)">{ent}</div>'))
  SLIDES[map_slide] = (body, "Appendix map: every backup slide and the main-deck slide it supports (arrow).", "light")


# ------------------------------------------------------------------ page

CSS = """
@font-face { font-family: 'Inter Tight'; src: url(data:font/ttf;base64,__DISP__) format('truetype'); font-weight: 100 900; }
@font-face { font-family: 'Figtree'; src: url(data:font/ttf;base64,__FIG__) format('truetype'); font-weight: 300 900; }
* { box-sizing: border-box; }
html, body { margin: 0; background: #1a1a1a; height: 100%; overflow: hidden; }
.deck-slide { position: absolute; inset: 0; display: none; align-items: center; justify-content: center; }
.deck-slide.active { display: flex; }
.slide { width: 1920px; height: 1080px; position: relative; overflow: hidden; flex: none; font-family: 'Figtree', Arial, sans-serif;
  color: __INK__; background: __PAPER__; }
.slide.dark { background: __DARK__; color: __CREAM__; }
.box { position: absolute; overflow: hidden; }
.box > svg { display: block; }
.box > svg:not(.bare), .vis svg:not(.bare) { background: __CARD__; border-radius: 22px; padding: 18px 24px; }
.box > svg:not(.bare), .vis svg:not(.bare), .card, .stat, .learn, .panel, .tbl, .dec, .lesson, .rstep, .task, .agenda, .q, .factor, .cmp, .sh-r, .take, .layer, .formula, .srow, .ri, .pr { box-shadow: inset 0 0 0 2px __RULE__; }
.kicker { position: absolute; font-size: 24px; font-weight: 700; letter-spacing: .16em; text-transform: uppercase; display: flex; align-items: center; gap: 14px; }
.dot { width: 18px; height: 18px; border-radius: 50%; display: inline-block; }
.title { position: absolute; margin: 0; font-family: 'Inter Tight', 'Figtree', Arial, sans-serif; font-weight: 700; letter-spacing: -0.01em; font-size: 62px; line-height: 1.1; white-space: nowrap; overflow: hidden; color: __DARK__; }
.track { position: absolute; top: 80px; width: 40px; height: 8px; border-radius: 4px; }
.source { position: absolute; top: 1004px; font-size: 20px; color: __SOURCE__; white-space: nowrap; overflow: hidden; }
.hero { position: absolute; margin: 0; font-family: 'Inter Tight', 'Figtree', Arial, sans-serif; font-weight: 700; letter-spacing: -0.01em; font-size: 96px; line-height: 1.04; color: __CREAM__; }
.div-title { position: absolute; margin: 0; font-family: 'Inter Tight', 'Figtree', Arial, sans-serif; font-weight: 700; letter-spacing: -0.01em; font-size: 120px; line-height: 1.0; }
.div-sub { position: absolute; font-size: 36px; line-height: 1.35; }
.author { position: absolute; font-size: 30px; line-height: 1.5; }
.bignum { position: absolute; font-family: 'Inter Tight', 'Figtree', Arial, sans-serif; font-weight: 700; letter-spacing: -0.01em; font-size: 300px; line-height: 1; color: transparent; }
.circle { position: absolute; border-radius: 50%; }
.statement { position: absolute; font-family: 'Inter Tight', 'Figtree', Arial, sans-serif; font-weight: 700; letter-spacing: -0.01em; font-size: 84px; line-height: 1.12; }
.agenda { position: absolute; overflow: hidden; background: __CARD__; border-radius: 26px; padding: 40px 34px; }
.ag-n { font-family: 'Inter Tight', 'Figtree', Arial, sans-serif; font-weight: 700; letter-spacing: -0.01em; font-size: 76px; line-height: 1; }
.ag-t { font-family: 'Inter Tight', 'Figtree', Arial, sans-serif; font-weight: 700; letter-spacing: -0.01em; font-size: 40px; margin: 44px 0 22px; line-height: 1.1; }
.ag-m { position: absolute; left: 34px; bottom: 34px; font-size: 24px; font-weight: 800; letter-spacing: .12em; text-transform: uppercase; color: __MUTED__; }
.ag-d { font-size: 30px; line-height: 1.42; color: __MUTED__; }
.card { background: __CARD__; border-radius: 22px; padding: 28px 32px; height: 100%; position: relative; }
.card-t { font-family: 'Inter Tight', 'Figtree', Arial, sans-serif; font-weight: 700; letter-spacing: -0.01em; font-size: 34px; line-height: 1.15; margin-bottom: 14px; }
.tag + .card-t { padding-right: 150px; }
.card-b { font-size: 26px; line-height: 1.42; }
.tag { position: absolute; top: 28px; right: 26px; font-size: 18px; font-weight: 700; letter-spacing: .1em; text-transform: uppercase; padding: 6px 14px; border-radius: 20px; }
.learn { background: __CARD__; border-radius: 18px; padding: 24px 32px; height: 100%; display: flex; flex-direction: column; justify-content: center; }
.learn-l { font-size: 20px; font-weight: 800; letter-spacing: .14em; text-transform: uppercase; margin-bottom: 10px; }
.learn-b { font-size: 28px; line-height: 1.42; }
.stat { background: __CARD__; border-radius: 22px; padding: 26px 32px; height: 100%; display: flex; flex-direction: column; justify-content: center; }
.stat-big { font-family: 'Inter Tight', 'Figtree', Arial, sans-serif; font-weight: 700; letter-spacing: -0.01em; font-size: 70px; line-height: 1.05; white-space: nowrap; }
.stat-cap { font-size: 25px; line-height: 1.35; color: __MUTED__; margin-top: 12px; }
.vs { font-family: 'Inter Tight', 'Figtree', Arial, sans-serif; font-weight: 700; letter-spacing: -0.01em; font-size: 80px; color: __DARK__; height: 100%; display: flex; align-items: center; justify-content: center; }
.q { background: __CARD__; border-radius: 18px; padding: 24px 30px; height: 100%; font-size: 31px; line-height: 1.35; font-weight: 600; }
.q span { display: block; font-size: 20px; font-weight: 800; letter-spacing: .14em; text-transform: uppercase; color: __MUTED__; margin-bottom: 8px; }
.code { background: __INK__; border-radius: 18px; padding: 22px 30px; height: 100%; }
.code-l { font-size: 20px; font-weight: 800; letter-spacing: .1em; text-transform: uppercase; margin-bottom: 14px; }
.code pre { margin: 0; color: __CREAM__; font: 28px/1.5 'DejaVu Sans Mono', monospace; white-space: pre; }
.dec { background: __CARD__; border-radius: 22px; padding: 28px 32px; height: 100%; }
.dec-k { font-size: 20px; font-weight: 800; letter-spacing: .14em; text-transform: uppercase; color: __MUTED__; }
.dec-c { font-family: 'Inter Tight', 'Figtree', Arial, sans-serif; font-weight: 700; letter-spacing: -0.01em; font-size: 40px; line-height: 1.1; margin: 14px 0 6px; }
.dec-o { font-size: 23px; color: __MUTED__; font-style: italic; margin-bottom: 16px; }
.dec-w { font-size: 26px; line-height: 1.4; }
.step { position: absolute; height: 58px; border-radius: 29px; display: flex; align-items: center; justify-content: center; font-size: 24px; font-weight: 600; }
.panel { background: __CARD__; border-radius: 20px; padding: 20px 28px; display: flex; flex-direction: column; justify-content: center; }
.pl { font-size: 19px; font-weight: 800; letter-spacing: .14em; text-transform: uppercase; color: __MUTED__; margin-bottom: 8px; }
.pb { font-size: 24px; line-height: 1.38; }
.vis { display: flex; align-items: center; justify-content: center; }
.lesson { background: __CARD__; border-radius: 22px; padding: 30px 32px; height: 100%; }
.lesson-n { width: 64px; height: 64px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 30px; font-weight: 800; }
.lesson-k { font-family: 'Inter Tight', 'Figtree', Arial, sans-serif; font-weight: 700; letter-spacing: -0.01em; font-size: 40px; margin: 22px 0 10px; }
.lesson-t { font-size: 27px; line-height: 1.4; }
.factor { background: __CARD__; border-radius: 22px; padding: 28px 32px; height: 100%; }
.f-t { font-family: 'Inter Tight', 'Figtree', Arial, sans-serif; font-weight: 700; letter-spacing: -0.01em; font-size: 32px; margin-bottom: 14px; }
.f-r { display: flex; justify-content: space-between; align-items: baseline; font-size: 26px; padding: 14px 0; border-bottom: 2px solid __RULE__; }
.f-r b { font-size: 30px; }
.f-n { font-size: 20px; color: __MUTED__; margin-top: 16px; line-height: 1.4; }
.ph { border: 4px dashed __SKY__; border-radius: 24px; height: 100%; padding: 34px; background: __CARD__; display: flex; flex-direction: column; justify-content: center; }
.ph-l { font-family: 'Inter Tight', 'Figtree', Arial, sans-serif; font-weight: 700; letter-spacing: -0.01em; font-size: 36px; margin-bottom: 18px; }
.ph-i { font-size: 26px; color: __MUTED__; line-height: 1.6; }
.ph-n { margin-top: 18px; font-size: 20px; font-weight: 800; letter-spacing: .14em; text-transform: uppercase; color: __BLUE__; }
.cmp { background: __CARD__; border-radius: 22px; padding: 22px 30px; height: 100%; display: flex; flex-direction: column; justify-content: center; }
.cmp-h, .cmp-r { display: grid; grid-template-columns: 34% 33% 33%; align-items: center; font-size: 24px; padding: 13px 0; border-bottom: 2px solid __RULE__; }
.cmp-h b { font-size: 20px; letter-spacing: .08em; text-transform: uppercase; }
.cmp-r span { color: __MUTED__; } .cmp-r i { color: __BLUE__; font-style: normal; font-weight: 700; }
.field { background: __INK__; color: __PAPER__; border-radius: 24px; padding: 36px 36px; height: 100%; }
.field-h { font-family: 'Inter Tight', 'Figtree', Arial, sans-serif; font-weight: 700; letter-spacing: -0.01em; font-size: 42px; margin-bottom: 26px; color: __GOLD__; }
.field-i { font-size: 27px; line-height: 1.45; margin-bottom: 28px; }
.shared { display: flex; flex-direction: column; gap: 20px; height: 100%; }
.sh-r { background: __CARD__; border-radius: 20px; padding: 18px 26px; display: flex; gap: 22px; align-items: center; flex: 1; }
.sh-n { min-width: 52px; height: 52px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 26px; font-weight: 800; }
.sh-r b { font-size: 28px; } .sh-r span { font-size: 23px; color: __MUTED__; line-height: 1.38; }
.take { background: __CARD__; border-radius: 22px; padding: 30px 40px; display: grid; grid-template-columns: 1fr 1fr; gap: 22px 48px; height: 100%; align-content: center; }
.take div { font-size: 30px; line-height: 1.35; }
.appx { display: inline-block; padding: 4px 14px; border-radius: 16px; letter-spacing: .06em; }
.backref { position: absolute; max-width: 900px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; font-size: 21px; font-weight: 700;
  color: __MUTED__; border: 2px solid; border-radius: 22px; padding: 7px 18px; background: __CARD__; }
.tbl { background: __CARD__; border-radius: 22px; padding: 18px 30px; height: 100%; }
.tbl table { width: 100%; height: 100%; border-collapse: collapse; table-layout: fixed; }
.tbl thead tr { height: 1px; }
.tbl th { text-align: left; font-size: .78em; font-weight: 800; letter-spacing: .1em; text-transform: uppercase; color: __MUTED__; padding: 10px 12px 10px 0; border-bottom: 2px solid __INK__; }
.tbl td { padding: 11px 12px 11px 0; border-bottom: 2px solid __RULE__; line-height: 1.3; vertical-align: middle; }
.tbl tr:last-child td { border-bottom: none; }
.tbl tr.hl td { font-weight: 800; }
.mono { font-family: 'DejaVu Sans Mono', monospace; font-size: .86em; }
.layer { background: __CARD__; border-radius: 20px; height: 100%; display: flex; align-items: center; gap: 24px; padding: 0 28px; border-left: 12px solid; }
.layer-l { width: 270px; flex: none; font-family: 'Inter Tight', 'Figtree', Arial, sans-serif; font-weight: 700; letter-spacing: -0.01em; font-size: 28px; line-height: 1.15; }
.layer-l span { display: block; font-family: 'Figtree', Arial, sans-serif; font-weight: 400; letter-spacing: 0; font-size: 19px; color: __MUTED__; margin-top: 4px; }
.pills { display: flex; flex-wrap: wrap; gap: 10px; }
.pill { font-size: 23px; font-weight: 700; padding: 8px 18px; border-radius: 24px; background: __PAPER__; border: 2px solid __RULE__; }
.rstep { background: __CARD__; border-radius: 22px; padding: 24px 28px; height: 100%; display: flex; flex-direction: column; }
.rstep-h { display: flex; align-items: center; gap: 16px; font-family: 'Inter Tight', 'Figtree', Arial, sans-serif; font-weight: 700; letter-spacing: -0.01em; font-size: 34px; margin-bottom: 12px; }
.rstep-h b { width: 50px; height: 50px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-family: 'Figtree', Arial, sans-serif; font-size: 25px; }
.rstep-d { font-size: 24px; line-height: 1.38; flex: 1; }
.rstep-s { font: 19px/1.4 'DejaVu Sans Mono', monospace; color: __MUTED__; border-top: 2px solid __RULE__; padding-top: 10px; }
.lh { font-family: 'Inter Tight', 'Figtree', Arial, sans-serif; font-weight: 700; letter-spacing: -0.01em; font-size: 32px; padding-left: 80px; }
.lh span { font-family: 'Figtree', Arial, sans-serif; font-weight: 400; letter-spacing: 0; font-size: 22px; color: __SOURCE__; margin-left: 10px; }
.formula { background: __INK__; color: __PAPER__; border-radius: 22px; height: 100%; padding: 28px 36px; display: flex; flex-direction: column; justify-content: center; }
.formula .fx { font-family: 'Inter Tight', 'Figtree', Arial, sans-serif; font-weight: 700; letter-spacing: -0.01em; font-size: 40px; line-height: 1.3; }
.formula .fn { font-size: 23px; color: __SKY__; margin-top: 14px; line-height: 1.4; }
.amap { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 6px 40px; height: 100%; align-content: start; }
.amap div { font-size: 21px; line-height: 1.3; padding: 7px 0; border-bottom: 2px solid __RULE__; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.amap b { display: inline-block; min-width: 54px; padding: 2px 8px; margin-right: 10px; border-radius: 10px; text-align: center; font-size: 18px; }
.amap i { font-style: normal; color: __MUTED__; margin-left: 6px; }
.task { background: __CARD__; border-radius: 22px; padding: 26px 30px; height: 100%; display: flex; flex-direction: column; }
.task-h { font-family: 'Inter Tight', 'Figtree', Arial, sans-serif; font-weight: 700; letter-spacing: -0.01em; font-size: 44px; line-height: 1; margin-bottom: 14px; }
.task-h span { display: block; font-family: 'Figtree', Arial, sans-serif; font-size: 20px; font-weight: 800; letter-spacing: .14em; text-transform: uppercase; color: __MUTED__; margin-top: 10px; }
.task-r { font-size: 23px; line-height: 1.32; padding: 11px 0; border-top: 2px solid __RULE__; flex: 1; display: flex; flex-direction: column; justify-content: center; }
.task-r > span { font-size: 17px; font-weight: 800; letter-spacing: .12em; text-transform: uppercase; color: __MUTED__; margin-bottom: 3px; }
.cmp4 .cmp-h, .cmp4 .cmp-r { grid-template-columns: 22% 28% 22% 28%; }
.hh { height: 100%; display: flex; align-items: flex-end; gap: 14px; padding-bottom: 10px; border-bottom: 3px solid __RULE__; white-space: nowrap;
  font-family: 'Inter Tight', 'Figtree', Arial, sans-serif; font-weight: 700; letter-spacing: -0.01em; font-size: 32px; line-height: 1.1; color: __DARK__; }
.hh span { font-family: 'Figtree', Arial, sans-serif; font-weight: 400; letter-spacing: 0; font-size: 21px; color: __MUTED__; }
.vsep { position: absolute; width: 2px; background: __RULE__; }
.srow { background: __CARD__; border-radius: 18px; padding: 10px 24px; height: 100%; display: flex; align-items: center; gap: 20px; }
.srow-b { font-family: 'Inter Tight', 'Figtree', Arial, sans-serif; font-weight: 700; letter-spacing: -0.01em; font-size: 40px; line-height: 1; white-space: nowrap; flex: none; min-width: 110px; }
.srow-c { font-size: 20px; line-height: 1.3; color: __MUTED__; }
.ri { background: __CARD__; border-radius: 16px; padding: 8px 24px; height: 100%; display: flex; align-items: center; font-size: 20px; line-height: 1.34; }
.ri b { font-weight: 800; }
.sh { height: 100%; display: flex; align-items: center; gap: 14px; border-bottom: 3px solid __RULE__; padding-bottom: 6px; }
.sh b { width: 38px; height: 38px; border-radius: 50%; background: __DARK__; color: #ffffff; display: flex; align-items: center; justify-content: center; font-size: 20px; font-weight: 800; flex: none; }
.sh i { font-style: normal; font-size: 18px; font-weight: 800; letter-spacing: .14em; text-transform: uppercase; color: __MUTED__; }
.sh span { font-family: 'Inter Tight', 'Figtree', Arial, sans-serif; font-weight: 700; letter-spacing: -0.01em; font-size: 30px; color: __DARK__; white-space: nowrap; }
.pr { background: __CARD__; border-radius: 16px; padding: 6px 22px; height: 100%; display: flex; align-items: center; font-size: 20px; line-height: 1.32; }
.pr .prl { font-size: 15px; font-weight: 800; letter-spacing: .14em; text-transform: uppercase; margin-right: 10px; }
.card.sm { padding: 18px 24px; } .card.sm .card-t { font-size: 26px; margin-bottom: 8px; } .card.sm .card-b { font-size: 21px; line-height: 1.38; }
.learn.sm { padding: 16px 26px; } .learn.sm .learn-l { font-size: 17px; margin-bottom: 6px; } .learn.sm .learn-b { font-size: 21px; line-height: 1.38; }
.stat.sm { padding: 14px 26px; } .stat.sm .stat-big { font-size: 46px; } .stat.sm .stat-cap { font-size: 20px; margin-top: 6px; }
.code.sm { padding: 14px 26px; } .code.sm .code-l { font-size: 17px; margin-bottom: 8px; } .code.sm pre { font-size: 23px; line-height: 1.42; }
.q.sm { font-size: 26px; padding: 16px 26px; } .q.sm span { font-size: 17px; margin-bottom: 4px; }
.tbl.sm { padding: 6px 22px; } .tbl.sm th { padding: 4px 10px 4px 0; } .tbl.sm td { padding: 4px 10px 4px 0; }
.task.sm { padding: 18px 24px; } .task.sm .task-h { font-size: 34px; margin-bottom: 4px; display: flex; align-items: baseline; gap: 14px; }
.task.sm .task-h span { display: inline; margin-top: 0; font-size: 17px; }
.task.sm .task-r { display: block; font-size: 20px; line-height: 1.3; padding: 7px 0; }
.task.sm .task-r > span { display: inline; font-size: 14px; margin-right: 10px; } .task.sm .task-r > div { display: inline; }
.layer.sm { gap: 18px; } .layer.sm .layer-l { font-size: 24px; width: 230px; } .layer.sm .layer-l span { font-size: 17px; }
.layer.sm .pill { font-size: 19px; padding: 5px 14px; } .layer.sm .pills { gap: 8px; }
.take.sm { padding: 20px 34px; gap: 14px 44px; } .take.sm div { font-size: 22px; line-height: 1.32; }
.rstep.sm { padding: 16px 18px; } .rstep.sm .rstep-h { font-size: 22px; gap: 10px; margin-bottom: 8px; line-height: 1.1; }
.rstep.sm .rstep-h b { width: 34px; height: 34px; font-size: 17px; flex: none; } .rstep.sm .rstep-d { font-size: 18px; line-height: 1.32; }
.rstep.sm .rstep-s { font-size: 14px; padding-top: 6px; }
.bet { background: __CREAM__; color: __INK__; box-shadow: inset 0 0 0 2px __GOLD__; border-radius: 22px; height: 100%; padding: 22px 34px; display: flex; flex-direction: column; justify-content: center; }
.bet-k { font-size: 20px; font-weight: 700; letter-spacing: .16em; text-transform: uppercase; display: flex; align-items: center; gap: 12px; }
.bet-s { font-family: 'Inter Tight', 'Figtree', Arial, sans-serif; font-weight: 700; letter-spacing: -0.01em; color: __DARK__; font-size: 36px; line-height: 1.15; margin: 10px 0 10px; }
.bet-b { font-size: 23px; line-height: 1.35; }
.source.two { top: 990px; font-size: 19px; line-height: 1.3; }
.build { opacity: 0; transition: opacity .3s ease; }
.build.on { opacity: 1; }
.notes { display: none; }
#notes-panel { position: fixed; left: 0; right: 0; bottom: 0; max-height: 34vh; overflow: auto; background: #ffffff; color: #111;
  font: 18px/1.5 Figtree, Arial, sans-serif; padding: 16px 28px; border-top: 4px solid __CORAL__; display: none; z-index: 10; white-space: pre-wrap; }
#counter { position: fixed; right: 18px; bottom: 12px; color: #bbb; font: 14px Figtree, Arial, sans-serif; z-index: 11; }
#qa { display: none; }
@media print {
  @page { size: 1920px 1080px; margin: 0; }
  html, body { background: none; overflow: visible; height: auto; }
  .deck-slide { position: relative; display: block !important; page-break-after: always; width: 1920px; height: 1080px; }
  .slide { transform: none !important; }
  .build { opacity: 1 !important; transition: none; }
  #notes-panel, #counter { display: none !important; }
}
"""

JS = """
const slides=[...document.querySelectorAll('.deck-slide')]; let cur=0, step=0;
const panel=document.getElementById('notes-panel'), counter=document.getElementById('counter');
function fit(){ const s=Math.min(innerWidth/1920, innerHeight/1080); document.querySelectorAll('.slide').forEach(e=>e.style.transform=`scale(${s})`); }
const builds=i=>[...slides[i].querySelectorAll('.build')];
const steps=i=>Math.max(0,...builds(i).map(b=>+b.dataset.step));
function paint(){ builds(cur).forEach(b=>b.classList.toggle('on',+b.dataset.step<=step)); const n=steps(cur);
  counter.textContent=`${cur+1} / ${slides.length}`+(n?` · ${step}/${n}`:''); }
function show(i,built){ cur=Math.max(0,Math.min(slides.length-1,i)); step=built?steps(cur):0; slides.forEach((s,j)=>s.classList.toggle('active',j===cur));
  panel.textContent=slides[cur].querySelector('.notes').textContent.trim(); paint();
  if(location.hash!=='#'+(cur+1)) history.replaceState(null,'','#'+(cur+1)); }
function next(){ if(step<steps(cur)){ step++; paint(); } else if(cur<slides.length-1) show(cur+1); }
function prev(){ if(step>0){ step--; paint(); } else if(cur>0) show(cur-1,true); }
addEventListener('keydown',e=>{ if(['ArrowRight','PageDown',' '].includes(e.key)) next();
  else if(['ArrowLeft','PageUp'].includes(e.key)) prev(); else if(e.key==='ArrowDown') show(cur+1); else if(e.key==='ArrowUp') show(cur-1);
  else if(e.key==='Home') show(0);
  else if(e.key==='End') show(slides.length-1); else if(e.key==='n'||e.key==='N') panel.style.display=panel.style.display==='block'?'none':'block';
  else if(e.key==='p'||e.key==='P') print(); });
addEventListener('click',e=>{ if(e.target.closest('#notes-panel')) return; next(); });
addEventListener('resize',fit); fit(); show((parseInt(location.hash.slice(1))||1)-1);
if(location.search.includes('qa')) document.fonts.ready.then(()=>{
  document.querySelectorAll('.slide').forEach(e=>e.style.transform='none');
  const out=[]; slides.forEach((s,i)=>{ s.classList.add('active'); const sr=s.querySelector('.slide').getBoundingClientRect();
    s.querySelectorAll('.box, .title, .statement, .div-title, .hero, .div-sub, .source, .stat-big').forEach(b=>{
      const isBox=b.classList.contains('box'); if((isBox&&b.scrollHeight>b.clientHeight+2)||b.scrollWidth>b.clientWidth+2) out.push(`slide ${i+1}: ${b.className.split(' ').slice(0,2).join('.')} overflows ${b.scrollWidth}x${b.scrollHeight} > ${b.clientWidth}x${b.clientHeight}`);
      const r=b.getBoundingClientRect(); if(r.right>sr.right+1||r.bottom>sr.bottom+1) out.push(`slide ${i+1}: ${b.className} leaves the slide`); });
    s.classList.remove('active'); });
  document.getElementById('qa').textContent='QA:'+(out.length?out.join(' | '):'clean'); show(cur);
});
"""


def page(fonts: Path) -> str:
  css = CSS
  for k, v in {"__PAPER__": PAPER, "__INK__": INK, "__MUTED__": MUTED, "__RULE__": RULE, "__CARD__": CARD, "__TEAL__": TEAL,
               "__CORAL__": CORAL, "__GOLD__": GOLD, "__SKY__": SKY, "__DARK__": DARK, "__CREAM__": CREAM, "__BLUE__": BLUE,
               "__SOURCE__": SOURCE}.items():
    css = css.replace(k, v)
  css = css.replace("__DISP__", base64.b64encode((fonts / "InterTight[wght].ttf").read_bytes()).decode())
  css = css.replace("__FIG__", base64.b64encode((fonts / "Figtree[wght].ttf").read_bytes()).decode())
  secs = []
  for i, (body, notes, theme) in enumerate(SLIDES, 1):
    cls = "slide dark" if theme == "dark" else "slide"
    secs.append(f'<section class="deck-slide" id="s{i}"><div class="{cls}">{body}</div><aside class="notes">{esc(notes)}</aside></section>')
  return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
          f'<title>Big-model quality, small-model price</title><style>{css}</style></head><body>' + "".join(secs) +
          f'<div id="notes-panel"></div><div id="counter"></div><div id="qa"></div><script>{JS}</script></body></html>')


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--out", type=Path, required=True)
  ap.add_argument("--fonts", type=Path, required=True)
  ap.add_argument("--notes-md", type=Path, default=None)
  args = ap.parse_args()
  build()
  args.out.write_text(page(args.fonts))
  if args.notes_md:
    lines = ["# Speaker notes: Big-model quality, small-model price", ""]
    for i, (body, notes, _) in enumerate(SLIDES, 1):
      m = re.search(r'class="(?:title|hero|div-title|statement)"[^>]*>(.*?)</(?:h1|h2|div)>', body, re.S)
      t = re.sub(r"<[^>]+>", " ", m.group(1)) if m else f"Slide {i}"
      lines += [f"## {i}. {html.unescape(' '.join(t.split()))}", "", notes.strip(), ""]
    args.notes_md.write_text("\n".join(lines))
  print(f"{len(SLIDES)} slides -> {args.out}")


if __name__ == "__main__":
  main()
