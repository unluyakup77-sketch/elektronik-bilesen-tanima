"""Otomatik AI desteği: CNN emin değilse AI'a sorulur ve sonuç açıkça "cnn+ai" olarak işaretlenir.
Gerçek Gemini çağrısı yapılmaz; ai_verify ve CNN olasılıkları sahte değerlerle değiştirilir (sunucu gerekmez)."""
import asyncio
import io

import pytest
from PIL import Image

pytestmark = pytest.mark.filterwarnings("ignore")


def _jpeg() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), "gray").save(buf, "JPEG")
    return buf.getvalue()


class _Upload:
    def __init__(self, data: bytes):
        self.data = data

    async def read(self, n: int) -> bytes:
        return self.data


def _run(monkeypatch, top_prob: float, key: str | None, ai_label: str = "sigorta"):
    from routers import scan

    classes = scan.model_classes()
    probs = [0.0] * len(classes)
    probs[classes.index("transformator")] = top_prob
    probs[classes.index("role")] = (1 - top_prob) / 2
    probs[classes.index("buton")] = (1 - top_prob) / 2
    monkeypatch.setattr(scan, "predict_bytes", lambda data, *a: probs)
    calls = []

    async def fake_ai(image, top3):
        calls.append(top3)
        return {"label": ai_label, "confidence": 0.9, "note": "test notu"}

    monkeypatch.setattr(scan, "ai_verify", fake_ai)
    if key:
        monkeypatch.setenv("GEMINI_API_KEY", key)
    else:
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    async def no_save(doc):
        return None

    monkeypatch.setattr(scan.db, "save_prediction", no_save)
    monkeypatch.setattr(scan, "UPLOADS_DIR", scan.Path(__import__("tempfile").mkdtemp()))
    pred = asyncio.run(scan.predict(_Upload(_jpeg()), double_check=False))
    return pred, calls


def test_confident_cnn_does_not_call_ai(monkeypatch):
    pred, calls = _run(monkeypatch, 0.9, key="x")
    assert calls == [] and pred.source == "cnn" and pred.label == "transformator"


def test_unsure_cnn_calls_ai_and_is_labeled(monkeypatch):
    pred, calls = _run(monkeypatch, 0.4, key="x")
    assert len(calls) == 1
    assert pred.source == "cnn+ai" and pred.ai_used and pred.label == "sigorta"
    assert pred.ai_note.startswith("CNN emin olmadığı için")


def test_unsure_cnn_without_key_stays_cnn(monkeypatch):
    pred, calls = _run(monkeypatch, 0.4, key=None)
    assert calls == [] and pred.source == "cnn" and pred.label == "transformator"
