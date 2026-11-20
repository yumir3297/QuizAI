"""OCR-first router: decide whether a screenshot needs the vision model.

The key question is NOT "can OCR read it?" but "does answering require visual
understanding?". OCR can read a graph's caption perfectly yet be useless for
answering "which curve is f(x)?".

Phase-1 router uses three cheap, pre-LLM signals (benchmark-validated):

  1. OCR quality   — very low mean confidence → text is unreliable → vision.
  2. Visual deps   — referential phrases ("the figure above", "graph shows")
                     or visual-noun keywords (graph/diagram/circuit/chart/…).
                     This is the most important signal.
  3. Math damage   — rendered super/subscripts and symbols (2³, √, ∫, ½) get
                     silently corrupted by OCR at HIGH confidence (2³ → "23"),
                     so symbol-heavy text routes to vision regardless of score.

The text LLM also self-reports ``requires_visual_context`` per question as a
backstop for subtle cases the regex misses (handled by the worker).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Referential phrases — the question points at something only visible in the image.
_REFERENTIAL = re.compile(
    r"\b(?:figure|fig\.|graph|diagram|chart|plot|circuit|image|picture|"
    r"shown\s+(?:above|below)|(?:above|below|following)\s+(?:figure|graph|diagram|"
    r"image|chart|table)|refer\s+to|as\s+shown|in\s+the\s+(?:figure|diagram|graph|"
    r"image|picture|map))\b",
    re.IGNORECASE,
)

# Visual-content nouns that usually demand seeing the picture.
_VISUAL_NOUN = re.compile(
    r"\b(?:graph|diagram|flowchart|geometry|circuit|histogram|scatter|"
    r"number\s+line|coordinate\s+plane|venn)\b",
    re.IGNORECASE,
)

# OCR often drops spaces ("Accordingtothechart"), so word-boundary regexes miss
# the visual keywords. Matching against a space-stripped copy catches those.
# A false positive only costs the OCR hint — the image remains authoritative.
_COMPACT_VISUAL = (
    "chart",
    "graph",
    "diagram",
    "figure",
    "circuit",
    "flowchart",
    "histogram",
    "scatter",
    "venn",
    "geometry",
)

# Chinese questions reference visuals without spaces; \b never matches here.
_CJK_VISUAL = (
    "如图",
    "下图",
    "上图",
    "下表",
    "上表",
    "图中",
    "图表",
    "图形",
    "图像",
    "曲线",
    "示意",
    "电路",
    "表格",
    "统计图",
    "柱状图",
    "折线图",
    "饼图",
)

# Rendered-math glyphs OCR commonly *corrupts*: superscripts/subscripts (e.g. 2^3
# misread as 23), roots, integrals, sums, vulgar fractions. Plain operators and
# symbols (times, divide, minus, plus, comparison and degree signs, pi, theta) are
# NOT included -- OCR reads those fine, so simple arithmetic stays on the text path.
_MATH_GLYPHS = "√∫∑∏⅓⅔⅛⅜⅝½¼¾²³⁰¹⁴⁵⁶⁷⁸⁹₀₁₂₃₄₅₆₇₈₉"
_MATH_GLYPH_RE = re.compile(f"[{re.escape(_MATH_GLYPHS)}]")


@dataclass
class RouteDecision:
    use_vision: bool
    reason: str


def decide_route(
    text: str,
    mean_confidence: float,
    char_count: int,
    conf_floor: float = 0.55,
) -> RouteDecision:
    """Pre-LLM routing on the whole OCR result. True ⇒ go straight to vision."""
    if char_count < 3:
        return RouteDecision(True, "no usable text from OCR")
    if mean_confidence < conf_floor:
        return RouteDecision(True, f"low OCR confidence ({mean_confidence:.2f})")
    if _references_visual(text):
        return RouteDecision(True, "references a figure/graph/diagram")
    if _math_is_heavy(text):
        return RouteDecision(True, "rendered math glyphs (OCR-unreliable)")
    return RouteDecision(False, "plain text — OCR/text path")


def _references_visual(text: str) -> bool:
    """True if the text points at something only visible in the image."""
    if _REFERENTIAL.search(text) or _VISUAL_NOUN.search(text):
        return True
    compact = re.sub(r"\s+", "", text).lower()
    if any(word in compact for word in _COMPACT_VISUAL):
        return True
    return any(word in text for word in _CJK_VISUAL)


def _math_is_heavy(text: str) -> bool:
    """True if rendered-math glyphs appear that OCR likely mangled.

    Any superscript/subscript/root/integral/fraction glyph is enough: OCR turns
    ``x²`` into ``x2`` at high confidence, and a single corrupted symbol can
    change the answer. Degree signs are not in the list, so "90°" stays on the
    text path.
    """
    return bool(_MATH_GLYPH_RE.search(text))
