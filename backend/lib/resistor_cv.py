"""
Klasik görüntü işleme ile direnç renk bandı okuma (OpenCV + NumPy, yapay zekâ YOK).

Adımlar:
 1. Arka plan rengi kenar piksellerinden tahmin edilir, beyaz dengesi bu renge göre yapılır.
 2. Arka plandan farklı pikseller maskelenir → en büyük kontur = direnç (bacaklarla birlikte).
 3. minAreaRect ile direncin açısı bulunur, görüntü döndürülerek direnç yatay hâle getirilir.
 4. Bacaklar kesilir: maske kalınlığı gövdenin %55'inden az olan sütunlar atılır.
 5. Gövdenin orta şeridinden sütun bazında medyan renk profili çıkarılır.
 6. Gövde renginden (bej/mavi) belirgin şekilde farklı olan ardışık sütunlar = renk bantları.
 7. Her bant Lab renk uzayında referans renklere olan uzaklığa göre sınıflandırılır; rol kuralları
    (rakam bantları altın/gümüş olamaz, tolerans bandı belirli renklerdir) ve iki okuma yönü denenerek
    en düşük toplam uzaklığa sahip geçerli kombinasyon seçilir.
"""
from __future__ import annotations


import cv2
import numpy as np

from lib.resistor import DIGITS, MULTIPLIERS, TOLERANCES

# Tipik direnç fotoğraflarından ölçülmüş referans renkler (RGB)
REF_RGB: dict[str, tuple[int, int, int]] = {
    "black": (30, 30, 30),
    "brown": (115, 65, 40),
    "red": (200, 35, 40),
    "orange": (240, 120, 30),
    "yellow": (235, 205, 40),
    "green": (40, 140, 60),
    "blue": (30, 90, 190),
    "violet": (130, 60, 150),
    "grey": (125, 125, 125),
    "white": (235, 235, 235),
    "gold": (185, 150, 75),
    "silver": (190, 190, 195),
}
_NAMES = list(REF_RGB)
_REF_LAB = cv2.cvtColor(np.array([[REF_RGB[n][::-1] for n in _NAMES]], np.uint8), cv2.COLOR_BGR2LAB)[0].astype(np.float32)


def _lab(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.clip(bgr, 0, 255).astype(np.uint8).reshape(1, -1, 3), cv2.COLOR_BGR2LAB)[0].astype(np.float32)


def _distances(band_bgr: np.ndarray) -> np.ndarray:
    """Her bant için 12 referans renge Lab uzaklığı → (bant sayısı, 12)."""
    lab = _lab(band_bgr)
    w = np.array([0.6, 1.0, 1.0], np.float32)  # aydınlatma farkı (L) daha az önemli
    return np.sqrt((((lab[:, None, :] - _REF_LAB[None, :, :]) * w) ** 2).sum(-1))


def _runs(flags: np.ndarray) -> list[tuple[int, int]]:
    runs, start = [], None
    for i, f in enumerate(flags):
        if f and start is None:
            start = i
        elif not f and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(flags)))
    return runs


def _isolate_body(img: np.ndarray) -> np.ndarray | None:
    """Direnç gövdesini yatay ve bacaksız olarak döndürür (BGR)."""
    h, w = img.shape[:2]
    b = max(4, min(h, w) // 25)
    border = np.concatenate([img[:b].reshape(-1, 3), img[-b:].reshape(-1, 3), img[:, :b].reshape(-1, 3), img[:, -b:].reshape(-1, 3)])
    bg = np.median(border, axis=0)
    # beyaz dengesi: arka planı nötr griye çek
    gain = bg.mean() / np.maximum(bg, 1)
    img = np.clip(img.astype(np.float32) * gain, 0, 255).astype(np.uint8)
    bg = np.clip(bg * gain, 0, 255)

    lab = cv2.cvtColor(cv2.GaussianBlur(img, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    bg_lab = _lab(bg[None, :])[0]
    diff = np.linalg.norm(lab - bg_lab, axis=2)
    diff8 = np.clip(diff * 3, 0, 255).astype(np.uint8)
    _, mask = cv2.threshold(diff8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = max(3, min(h, w) // 60)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((k * 2, k * 2), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 0.01 * h * w:
        return None
    (cx, cy), (rw, rh), ang = cv2.minAreaRect(cnt)
    if rw < rh:
        ang += 90
        rw, rh = rh, rw
    M = cv2.getRotationMatrix2D((cx, cy), ang, 1.0)
    rot = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    rmask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST)
    x0, x1 = int(max(0, cx - rw / 2)), int(min(w, cx + rw / 2))
    y0, y1 = int(max(0, cy - rh / 2)), int(min(h, cy + rh / 2))
    if x1 - x0 < 20 or y1 - y0 < 6:
        return None
    sub, smask = rot[y0:y1, x0:x1], rmask[y0:y1, x0:x1] > 0

    # bacakları kes: gövde kalın, bacaklar ince
    thick = smask.sum(0).astype(np.float32)
    thick = np.convolve(thick, np.ones(5) / 5, mode="same")
    body_cols = _runs(thick >= 0.55 * thick.max())
    if not body_cols:
        return None
    bx0, bx1 = max(body_cols, key=lambda r: r[1] - r[0])
    if bx1 - bx0 < 20:
        return None
    rows = np.where(smask[:, bx0:bx1].mean(1) > 0.5)[0]
    if len(rows) < 4:
        return None
    ry0, ry1 = rows.min(), rows.max() + 1
    # orta şerit (parlama ve gölgeden kaçınmak için gövde yüksekliğinin ortadaki %40'ı)
    hh = ry1 - ry0
    sy0, sy1 = ry0 + int(hh * 0.30), ry0 + int(hh * 0.70) + 1
    # gövde uçlarındaki şişkin kısımları biraz kırp
    trim = int((bx1 - bx0) * 0.05)
    return sub[sy0:sy1, bx0 + trim : bx1 - trim]


def _best_combo(dist: np.ndarray) -> tuple[list[str], float, float] | None:
    """Rol kurallarına uyan, toplam renk uzaklığı en düşük kombinasyon (iki yön de denenir)."""
    n = dist.shape[0]
    if n not in (4, 5):
        return None
    best: tuple[list[str], float] | None = None
    for order in (list(range(n)), list(range(n))[::-1]):
        d = dist[order]
        roles = [list(DIGITS)] * (n - 2) + [list(MULTIPLIERS), list(TOLERANCES)]
        choice, total, margin = [], 0.0, 1e9
        for i, allowed in enumerate(roles):
            idx = [_NAMES.index(c) for c in allowed]
            ds = np.sort(d[i, idx])
            j = idx[int(np.argmin(d[i, idx]))]
            choice.append(_NAMES[j])
            total += float(d[i, j])
            margin = min(margin, float(ds[1] - ds[0]))  # en belirsiz bandın farkı
        # ilk rakam siyah olamaz (0 ile başlayan değer yok)
        if choice[0] == "black":
            total += 25.0
        if best is None or total < best[1]:
            best = (choice, total, margin)
    return best


def read_bands(data: bytes) -> dict:
    """Dönüş: {"bands": [...], "ok": bool, "score": ortalama renk uzaklığı, "reason": str}"""
    arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        return {"bands": [], "ok": False, "score": None, "reason": "Görsel okunamadı."}
    s = 600 / max(arr.shape[:2])
    if s < 1:
        arr = cv2.resize(arr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    body = _isolate_body(arr)
    if body is None:
        return {"bands": [], "ok": False, "score": None, "reason": "Direnç gövdesi görüntüde ayrıştırılamadı."}

    prof = np.median(body.astype(np.float32), axis=0)  # sütun başına medyan BGR
    prof = cv2.GaussianBlur(prof[None, :, :], (5, 1), 0)[0]
    lab = _lab(prof)
    body_lab = np.median(lab, axis=0)
    de = np.linalg.norm(lab - body_lab, axis=1)
    L = len(de)
    win = max(3, int(L * 0.02)) | 1
    de_s = np.convolve(de, np.ones(win) / win, mode="same")
    # gürültü seviyesi: gövde sütunlarının (en düşük %35) farkı; eşik bunun birkaç katı
    noise = float(np.percentile(de_s, 35))
    thr = max(14.0, noise * 3.0 + 6.0)
    merged: list[list[int]] = []
    for a, b in _runs(de_s > thr):  # yakın parçaları birleştir (ör. parlak altın bandın parçaları)
        if merged and a - merged[-1][1] <= max(1, L * 0.015):
            merged[-1][1] = b
        else:
            merged.append([a, b])
    merged = [r for r in merged if r[1] - r[0] >= max(2, L * 0.03)]
    if len(merged) not in (4, 5):
        return {"bands": [], "ok": False, "score": None, "reason": f"{len(merged)} bant bulundu (4 veya 5 bekleniyordu)."}
    band_bgr = np.array([np.median(prof[a + (b - a) // 4 : b - (b - a) // 4 or b], axis=0) for a, b in merged])
    combo = _best_combo(_distances(band_bgr))
    if combo is None:
        return {"bands": [], "ok": False, "score": None, "reason": "Bant renkleri eşleştirilemedi."}
    bands, total, margin = combo
    score = total / len(bands)
    # "Yanlış değer göstermektense hiç göstermemek" ilkesi: sadece renkler referanslara çok yakın ve
    # her bant için en yakın iki renk arasında belirgin fark varsa otomatik sonuç kabul edilir.
    ok = score < 19.0 and margin > 3.0
    return {"bands": bands, "ok": ok, "score": round(score, 1), "reason": "" if ok else "Renkler referanslara yeterince yakın değil."}
