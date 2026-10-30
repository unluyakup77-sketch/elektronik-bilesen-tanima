"""Hibrit karar: CNN tahmini + (gerekirse) Gemini görüntü modeli ile doğrulama."""
from __future__ import annotations

import base64
import json
import logging
import os
import re

import httpx

from lib.components_info import COMPONENTS, COMPONENTS_BY_KEY

logger = logging.getLogger(__name__)
AI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3-flash-preview")
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


class AINotConfigured(RuntimeError):
    """GEMINI_API_KEY tanımlı değil — çift kontrol yapılamaz, karar CNN'de kalır."""
AI_POLICY = "Karar CNN'indir. AI ikinci görüşü yalnızca CNN emin olmadığında (güven < %50) veya kullanıcı 'Çift kontrol'ü açtığında alınır ve sonuçta 'CNN + AI' olarak belirtilir."


def ai_configured() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY", "").strip())

_CLASS_LIST = "\n".join(f'- "{c["key"]}": {c["name"]} ({c["english_name"]})' for c in COMPONENTS)

SYSTEM_PROMPT = f"""Sen elektrik-elektronik bileşenlerini fotoğraftan tanıyan uzman bir mühendissin.
Fotoğraftaki ANA bileşeni dikkatle incele (gövde şekli, bacak sayısı, renk bantları, şerit, üzerindeki yazılar).
Etiket MUTLAKA şu anahtarlardan biri olmalı:
{_CLASS_LIST}
- "bilinmiyor": fotoğrafta bu listedeki bir elektronik bileşen yoksa veya karar verilemiyorsa.
Sadece JSON döndür, markdown kullanma:
{{"label": "<anahtar>", "confidence": <0-1 arası sayı>, "note": "<Türkçe, 1-2 cümle: hangi görsel ipuçlarına göre karar verdin>"}}"""


def _parse(raw: str) -> dict:
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        raise ValueError("JSON bulunamadı")
    data = json.loads(m.group(0))
    label = str(data.get("label", "bilinmiyor")).strip()
    if label not in COMPONENTS_BY_KEY:
        label = "bilinmiyor"
    conf = min(1.0, max(0.0, float(data.get("confidence", 0))))
    return {"label": label, "confidence": conf, "note": str(data.get("note", ""))[:400]}


async def ai_verify(image: bytes, cnn_top3: list[dict]) -> dict:
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise AINotConfigured()
    hint = ", ".join(f'{p["name"]} %{p["probability"] * 100:.1f}' for p in cnn_top3)
    body = {
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{
            "role": "user",
            "parts": [
                {"text": f"CNN modelimizin ilk 3 tahmini: {hint}. Bu tahminlere körü körüne güvenme, fotoğrafı kendin incele ve doğru etiketi ver."},
                {"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(image).decode("ascii")}},
            ],
        }],
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post(GEMINI_URL.format(model=AI_MODEL), json=body, headers={"x-goog-api-key": api_key})
        r.raise_for_status()
    parts = r.json()["candidates"][0]["content"]["parts"]
    return _parse("".join(p.get("text", "") for p in parts))

