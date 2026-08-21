"""Shared math-scoring helpers for answer equivalence.

Canonical home for ``_extract_boxed``, ``_normalize_math``, and
``_math_equiv``.  Every harness script that scores MATH / GSM8K
answers should import from here rather than maintaining a local copy.
"""

from __future__ import annotations

from fractions import Fraction
import re


def _extract_boxed(text: str) -> str | None:
    """Extract the last \\boxed{...} from a string, handling nested braces."""
    results = []
    i = 0
    while i < len(text):
        idx = text.find("\\boxed{", i)
        if idx == -1:
            break
        start = idx + len("\\boxed{")
        depth = 1
        j = start
        while j < len(text) and depth > 0:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
            j += 1
        if depth == 0:
            results.append(text[start : j - 1].strip())
        i = j
    return results[-1] if results else None


def _normalize_math(s: str | None) -> str:
    """Normalize a math answer string for comparison."""
    if s is None:
        return ""
    s = s.strip()
    if s.startswith("$") and s.endswith("$"):
        s = s[1:-1].strip()
    # Strip currency $ (e.g. "$18" -> "18")
    if re.match(r"^\$\d", s):
        s = s[1:]
    s = re.sub(r"\\text\s*\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mathrm\s*\{([^}]*)\}", r"\1", s)
    for cmd in [r"\left", r"\right", r"\displaystyle"]:
        s = s.replace(cmd, "")
    s = s.replace(r"\tfrac", r"\frac")
    s = s.replace(r"\dfrac", r"\frac")
    for unit in [
        " cm", " m", " km", " kg", " g", " s", " ms",
        " degrees", " degree", "\u00b0", " inches", " feet",
        " square units", " units", " dollars",
    ]:
        if s.lower().rstrip(".").endswith(unit):
            s = s[: len(s) - len(unit) - (1 if s.endswith(".") else 0)]
    s = re.sub(r"\s+", " ", s).strip()
    s = s.rstrip(".,")
    return s


def _parse_rational(s: str) -> Fraction | None:
    """Parse the deliberately small set of numeric answer forms we accept."""
    s = s.strip().replace(",", "")

    latex = re.fullmatch(
        r"([+-]?)\\frac\s*\{\s*([+-]?\d+)\s*\}\s*\{\s*([+-]?\d+)\s*\}",
        s,
    )
    if latex:
        try:
            value = Fraction(int(latex.group(2)), int(latex.group(3)))
            return -value if latex.group(1) == "-" else value
        except ZeroDivisionError:
            return None

    mixed = re.fullmatch(
        r"([+-]?\d+)\s*\\frac\s*\{\s*(\d+)\s*\}\s*\{\s*(\d+)\s*\}",
        s,
    )
    if mixed:
        try:
            whole = int(mixed.group(1))
            fraction = Fraction(int(mixed.group(2)), int(mixed.group(3)))
            return whole - fraction if whole < 0 else whole + fraction
        except ZeroDivisionError:
            return None

    slash = re.fullmatch(r"([+-]?\d+)\s*/\s*([+-]?\d+)", s)
    if slash:
        try:
            return Fraction(int(slash.group(1)), int(slash.group(2)))
        except ZeroDivisionError:
            return None

    if re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", s):
        return Fraction(s)
    return None


def _parse_interval(s: str) -> tuple[Fraction, bool, Fraction, bool] | None:
    """Recognize numeric interval notation and a chained x inequality only."""
    s = s.strip()
    membership = re.fullmatch(r"[xX]\s*\\in\s*(.+)", s)
    if membership:
        s = membership.group(1).strip()

    interval = re.fullmatch(r"([\[(])\s*(.+?)\s*,\s*(.+?)\s*([\])])", s)
    if interval:
        lower = _parse_rational(interval.group(2))
        upper = _parse_rational(interval.group(3))
        if lower is not None and upper is not None:
            return lower, interval.group(1) == "[", upper, interval.group(4) == "]"
        return None

    inequality = re.fullmatch(
        r"(.+?)\s*(\\leq?|≤|<)\s*[xX]\s*(\\leq?|≤|<)\s*(.+)",
        s,
    )
    if inequality:
        lower = _parse_rational(inequality.group(1))
        upper = _parse_rational(inequality.group(4))
        if lower is not None and upper is not None:
            return (
                lower,
                inequality.group(2) in (r"\le", r"\leq", "≤"),
                upper,
                inequality.group(3) in (r"\le", r"\leq", "≤"),
            )
    return None


def _strip_simple_answer_label(s: str) -> str:
    """Drop only a final-answer ``x =`` prefix, not arbitrary algebra."""
    labeled = re.fullmatch(r"[xX]\s*=\s*(.+)", s)
    if not labeled:
        return s
    answer = labeled.group(1).strip()
    if re.search(r"[xX]|\\(?:leq?|in)|[<>=≤]", answer):
        return s
    return answer


def _math_equiv(pred: str | None, gold: str | None) -> bool:
    """Check if two math answers are equivalent."""
    if pred is None or gold is None:
        return False
    p = _strip_simple_answer_label(_normalize_math(pred))
    g = _strip_simple_answer_label(_normalize_math(gold))
    if p == g:
        return True
    p_interval = _parse_interval(p)
    g_interval = _parse_interval(g)
    if p_interval is not None and g_interval is not None:
        return p_interval == g_interval
    p_rational = _parse_rational(p)
    g_rational = _parse_rational(g)
    if p_rational is not None and g_rational is not None:
        return p_rational == g_rational
    p_c = re.sub(r"\s*\\frac", r"\\frac", p)
    g_c = re.sub(r"\s*\\frac", r"\\frac", g)
    if p_c == g_c:
        return True
    try:
        pf = float(p.replace(",", ""))
        gf = float(g.replace(",", ""))
        return abs(pf - gf) < 1e-6
    except (ValueError, TypeError):
        pass
    mixed_pat = re.compile(r"^(\d+)\s*\\frac\s*\{(\d+)\}\s*\{(\d+)\}$")
    for s, other in [(p, g), (g, p)]:
        m = mixed_pat.match(s)
        if m:
            try:
                val = float(m.group(1)) + float(m.group(2)) / float(m.group(3))
                oval = float(other.replace(",", ""))
                if abs(val - oval) < 1e-6:
                    return True
            except (ValueError, ZeroDivisionError):
                pass
    frac_pat = re.compile(r"^\\frac\s*\{([^}]+)\}\s*\{([^}]+)\}$")
    for s, other in [(p, g), (g, p)]:
        m = frac_pat.search(s)
        if m:
            try:
                val = float(m.group(1)) / float(m.group(2))
                oval = float(other.replace(",", ""))
                if abs(val - oval) < 1e-6:
                    return True
            except (ValueError, ZeroDivisionError):
                pass
    return False
