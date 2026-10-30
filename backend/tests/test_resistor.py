"""Direnç renk kodu hesaplama (lib/resistor.py) ve OpenCV ile renk bandı okuma (lib/resistor_cv.py) testleri."""
import cv2
import numpy as np
import pytest

from lib import resistor
from lib.resistor_cv import REF_RGB, read_bands


# --- renk kodu hesaplama ----------------------------------------------------------------------------------------

@pytest.mark.parametrize(
    "bands, ohms, text, tol",
    [
        (["brown", "black", "red", "gold"], 1_000, "1 kΩ", 5),
        (["red", "red", "brown", "gold"], 220, "220 Ω", 5),
        (["yellow", "violet", "orange", "gold"], 47_000, "47 kΩ", 5),
        (["brown", "black", "yellow", "silver"], 100_000, "100 kΩ", 10),
        (["brown", "green", "green", "gold"], 1_500_000, "1.5 MΩ", 5),
        (["brown", "black", "blue", "gold"], 10_000_000, "10 MΩ", 5),
        (["green", "blue", "gold", "gold"], 5.6, "5.6 Ω", 5),
        (["blue", "grey", "silver", "gold"], 0.68, "0.68 Ω", 5),
        # 5 bantlı (hassas) dirençler
        (["brown", "black", "black", "brown", "brown"], 1_000, "1 kΩ", 1),
        (["orange", "orange", "black", "black", "brown"], 330, "330 Ω", 1),
        (["red", "violet", "black", "red", "red"], 27_000, "27 kΩ", 2),
    ],
)
def test_calculate_known_values(bands, ohms, text, tol):
    r = resistor.calculate(bands)
    assert r["ohms"] == pytest.approx(ohms)
    assert r["formatted"] == text
    assert r["tolerance_percent"] == tol


def test_min_max_follow_tolerance():
    r = resistor.calculate(["brown", "black", "red", "gold"])  # 1 kΩ ±%5
    assert r["min_ohms"] == pytest.approx(950)
    assert r["max_ohms"] == pytest.approx(1050)


def test_formula_uses_superscript_exponent():
    assert resistor.calculate(["brown", "black", "red", "gold"])["formula"] == "10 × 10² Ω"
    assert resistor.calculate(["green", "blue", "gold", "gold"])["formula"] == "56 × 10⁻¹ Ω"


@pytest.mark.parametrize(
    "bands",
    [
        [],
        ["brown", "black", "red"],  # 3 bant
        ["brown", "black", "red", "gold", "gold", "gold"],  # 6 bant
        ["gold", "black", "red", "gold"],  # rakam bandı altın olamaz
        ["brown", "silver", "red", "gold"],  # rakam bandı gümüş olamaz
        ["brown", "black", "pink", "gold"],  # bilinmeyen renk
        ["brown", "black", "red", "orange"],  # turuncu tolerans bandı değildir
    ],
)
def test_invalid_combinations_are_rejected(bands):
    assert not resistor.is_valid(bands)
    with pytest.raises(ValueError):
        resistor.calculate(bands)


def test_default_bands_are_valid():
    assert resistor.is_valid(resistor.DEFAULT_BANDS)


@pytest.mark.parametrize(
    "ohms, text",
    [(0.47, "0.47 Ω"), (47, "47 Ω"), (999, "999 Ω"), (1_000, "1 kΩ"), (4_700, "4.7 kΩ"),
     (2_200_000, "2.2 MΩ"), (1_000_000_000, "1 GΩ")],
)
def test_format_ohms(ohms, text):
    assert resistor.format_ohms(ohms) == text


def test_every_color_has_turkish_name():
    for c in set(resistor.DIGITS) | set(resistor.MULTIPLIERS) | set(resistor.TOLERANCES):
        assert c in resistor.COLOR_NAMES_TR


# --- OpenCV renk bandı okuma ------------------------------------------------------------------------------------

def _synthetic_resistor(bands: list[str], body=(225, 200, 160), bg=(250, 250, 250), angle: float = 0) -> bytes:
    """Bej gövdeli, bacaklı, renk bantlı yapay bir direnç fotoğrafı (JPEG) çizer."""
    img = np.full((300, 600, 3), bg[::-1], np.uint8)
    cv2.line(img, (40, 150), (560, 150), (150, 150, 150), 6)  # bacaklar
    cv2.rectangle(img, (170, 115), (430, 185), body[::-1], -1)  # gövde
    xs = np.linspace(195, 375, len(bands)) if len(bands) == 4 else np.linspace(190, 390, len(bands))
    if len(bands) == 4:
        xs[-1] = 395  # tolerans bandı ayrık durur
    for x, color in zip(xs, bands):
        cv2.rectangle(img, (int(x), 115), (int(x) + 16, 185), REF_RGB[color][::-1], -1)
    if angle:
        m = cv2.getRotationMatrix2D((300, 150), angle, 1.0)
        img = cv2.warpAffine(img, m, (600, 300), borderValue=bg[::-1])
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    assert ok
    return buf.tobytes()


def test_read_bands_returns_well_formed_result():
    r = read_bands(_synthetic_resistor(["brown", "black", "red", "gold"]))
    assert set(r) == {"bands", "ok", "score", "reason"}
    if r["ok"]:
        # otomatik okuma kabul edildiyse her zaman hesaplanabilir, geçerli bir kombinasyon olmalı
        assert resistor.is_valid(r["bands"])
        assert len(r["bands"]) in (4, 5)


@pytest.mark.parametrize("angle", [0, 12, -20])
def test_read_bands_never_returns_invalid_combination(angle):
    r = read_bands(_synthetic_resistor(["yellow", "violet", "orange", "gold"], angle=angle))
    if r["ok"]:
        assert resistor.is_valid(r["bands"])
    else:
        assert r["reason"]  # emin değilse sebebini söyler


def test_read_bands_rejects_non_image():
    r = read_bands(b"bu bir resim degil")
    assert r["ok"] is False and r["bands"] == [] and r["reason"]


def test_read_bands_rejects_image_without_resistor():
    img = np.full((240, 320, 3), 255, np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    r = read_bands(buf.tobytes())
    assert r["ok"] is False and r["reason"]
