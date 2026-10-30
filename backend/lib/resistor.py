"""Direnç renk kodu hesaplama (4 ve 5 bantlı dirençler)."""
from __future__ import annotations

import math

DIGITS: dict[str, int] = {
    "black": 0, "brown": 1, "red": 2, "orange": 3, "yellow": 4,
    "green": 5, "blue": 6, "violet": 7, "grey": 8, "white": 9,
}
MULTIPLIERS: dict[str, float] = {
    "black": 1, "brown": 10, "red": 100, "orange": 1e3, "yellow": 1e4, "green": 1e5,
    "blue": 1e6, "violet": 1e7, "grey": 1e8, "white": 1e9, "gold": 0.1, "silver": 0.01,
}
TOLERANCES: dict[str, float] = {
    "brown": 1, "red": 2, "green": 0.5, "blue": 0.25, "violet": 0.1, "grey": 0.05, "gold": 5, "silver": 10,
}
COLOR_NAMES_TR: dict[str, str] = {
    "black": "siyah", "brown": "kahverengi", "red": "kırmızı", "orange": "turuncu", "yellow": "sarı",
    "green": "yeşil", "blue": "mavi", "violet": "mor", "grey": "gri", "white": "beyaz",
    "gold": "altın", "silver": "gümüş",
}
ALL_COLORS = list(COLOR_NAMES_TR.keys())
DEFAULT_BANDS = ["brown", "black", "red", "gold"]  # 1 kΩ ±%5


def _sup(n: int) -> str:
    return str(n).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))


def format_ohms(ohms: float) -> str:
    for limit, unit in ((1e9, "GΩ"), (1e6, "MΩ"), (1e3, "kΩ")):
        if ohms >= limit:
            return f"{ohms / limit:.3f}".rstrip("0").rstrip(".") + f" {unit}"
    return f"{ohms:.2f}".rstrip("0").rstrip(".") + " Ω"


def is_valid(bands: list[str]) -> bool:
    if len(bands) not in (4, 5):
        return False
    *digits, mult, tol = bands
    return all(d in DIGITS for d in digits) and mult in MULTIPLIERS and tol in TOLERANCES


def calculate(bands: list[str]) -> dict:
    """bands: okuma sırasına göre renkler. 4 bant: d1 d2 çarpan tolerans · 5 bant: d1 d2 d3 çarpan tolerans."""
    if not is_valid(bands):
        raise ValueError("Geçersiz bant kombinasyonu.")
    *digits, mult, tol = bands
    base = int("".join(str(DIGITS[d]) for d in digits))
    ohms = round(base * MULTIPLIERS[mult], 4)
    t = TOLERANCES[tol]
    return {
        "ohms": ohms,
        "formatted": format_ohms(ohms),
        "tolerance_percent": t,
        "min_ohms": round(ohms * (1 - t / 100), 4),
        "max_ohms": round(ohms * (1 + t / 100), 4),
        "formula": f"{base} × 10{_sup(round(math.log10(MULTIPLIERS[mult])))} Ω",
    }
