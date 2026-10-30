import asyncio
import base64
import io
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, ImageOps, UnidentifiedImageError

from lib.components_info import COMPONENTS, COMPONENTS_BY_KEY
from lib import resistor
from lib import db
from lib.resistor_cv import read_bands
from lib.verifier import AI_MODEL, AI_POLICY, AINotConfigured, ai_configured, ai_verify
from ml.inference import LIVE_VIEWS, model_available, model_classes, predict_bytes
from models.prediction import (
    ClassProb,
    ComponentInfo,
    FeedbackIn,
    FeedbackResult,
    ModelInfo,
    Prediction,
    ResistorReading,
    LivePrediction,
    ResistorValue,
    Sample,
)

router = APIRouter()
logger = logging.getLogger(__name__)
ML_DIR = Path(__file__).resolve().parent.parent / "ml"
SAMPLES_DIR = ML_DIR / "samples"
# Taranan fotoğrafların kendisi (geri bildirimde eğitim setine kopyalanır) ve kullanıcı düzeltmeleri
UPLOADS_DIR = Path(__file__).resolve().parent.parent / "data" / "uploads"
USER_DATASET = ML_DIR / "dataset_user"
MAX_BYTES = 10 * 1024 * 1024
UNKNOWN_NAME = "Bileşen değil"
AUTO_AI_BELOW = 0.5  # CNN güveni bunun altındaysa (ve anahtar varsa) otomatik AI desteği


def _thumbnail(img: Image.Image) -> str:
    t = img.copy()
    t.thumbnail((320, 320))
    buf = io.BytesIO()
    t.save(buf, format="JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _normalized_jpeg(img: Image.Image) -> bytes:
    t = img.copy()
    t.thumbnail((1024, 1024))
    buf = io.BytesIO()
    t.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _display_name(key: str) -> str:
    """Sınıf anahtarının Türkçe adı; "bilinmiyor" (bileşen değil) sınıfının bilgi kartı yoktur."""
    return COMPONENTS_BY_KEY[key]["name"] if key in COMPONENTS_BY_KEY else UNKNOWN_NAME


@router.get("/components", response_model=list[ComponentInfo])
async def list_components():
    return COMPONENTS


@router.post("/predict", response_model=Prediction)
async def predict(file: UploadFile = File(...), double_check: bool = Form(False)):
    data = await file.read(MAX_BYTES + 1)
    if not data:
        raise HTTPException(400, "Boş dosya yüklendi.")
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Dosya çok büyük (en fazla 10 MB).")
    try:
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(415, "Geçerli bir görsel dosyası değil (JPG, PNG, WEBP yükleyin).")
    if not model_available():
        raise HTTPException(503, "CNN modeli henüz eğitilmedi.")

    jpeg = _normalized_jpeg(img)
    classes = model_classes()
    probs = await asyncio.to_thread(predict_bytes, jpeg)
    order = sorted(range(len(classes)), key=lambda i: -probs[i])
    top3 = [{"key": classes[i], "name": _display_name(classes[i]), "probability": round(probs[i], 4)} for i in order[:3]]
    cnn_label, cnn_conf = top3[0]["key"], top3[0]["probability"]

    label, conf, source = cnn_label, cnn_conf, "cnn"
    ai_used, ai_label, ai_conf, ai_note, agreement = False, None, None, None, None
    # Karar varsayılan olarak CNN'indir. AI ikinci görüşü iki durumda alınır (sonuçta her zaman açıkça belirtilir):
    #  1. kullanıcı "Çift kontrol"ü açtıysa
    #  2. CNN emin değilse (güven < AUTO_AI_BELOW) ve GEMINI_API_KEY tanımlıysa — otomatik AI desteği
    auto_ai = not double_check and cnn_conf < AUTO_AI_BELOW and ai_configured()
    if double_check or auto_ai:
        try:
            ai = await ai_verify(jpeg, top3)
            ai_used, ai_label, ai_conf = True, ai["label"], ai["confidence"]
            reason = f"CNN emin olmadığı için (%{cnn_conf * 100:.0f}) AI desteği alındı. " if auto_ai else ""
            ai_note = reason + ai["note"]
            agreement = ai_label == cnn_label
            source = "cnn+ai"
            if agreement:
                conf = max(cnn_conf, ai_conf)
            else:
                label, conf = ai_label, ai_conf
        except AINotConfigured:
            ai_note = "Çift kontrol için GEMINI_API_KEY ayarlanmamış (backend/.env); sonuç yalnızca CNN tahminine dayanıyor."
        except Exception as e:  # AI ulaşılamazsa CNN sonucu kullanılır
            ai_note = "AI doğrulaması şu an yapılamadı; sonuç yalnızca CNN tahminine dayanıyor."
            logger.warning("AI doğrulama hatası: %s", e)

    is_component = label != "bilinmiyor"
    resistor_reading = await _read_resistor(jpeg) if label == "direnc" else None
    pred = Prediction(
        id=str(uuid.uuid4()),
        created_at=datetime.now(timezone.utc),
        label=label,
        name=COMPONENTS_BY_KEY[label]["name"] if is_component else "Tanımlanamadı",
        confidence=round(conf, 4),
        is_component=is_component,
        source=source,
        cnn_label=cnn_label,
        cnn_confidence=cnn_conf,
        cnn_top3=[ClassProb(**p) for p in top3],
        ai_used=ai_used,
        ai_label=ai_label,
        ai_confidence=ai_conf,
        ai_note=ai_note,
        agreement=agreement,
        thumbnail=_thumbnail(img),
        info=ComponentInfo(**COMPONENTS_BY_KEY[label]) if is_component else None,
        resistor=resistor_reading,
    )
    await db.save_prediction(pred.model_dump(mode="json"))
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    (UPLOADS_DIR / f"{pred.id}.jpg").write_bytes(jpeg)
    return pred


@router.post("/predictions/{id}/feedback", response_model=FeedbackResult)
async def prediction_feedback(id: str, body: FeedbackIn):
    """Kullanıcı doğru etiketi bildirir: fotoğraf ml/dataset_user/<etiket>/ altına kopyalanır.
    Bu fotoğraflar bir sonraki ince ayarda (python -m ml.train --init ...) eğitime girer."""
    classes = model_classes()
    if body.label not in classes:
        raise HTTPException(400, "Geçersiz etiket.")
    src = UPLOADS_DIR / f"{id}.jpg"
    if not src.exists():
        raise HTTPException(404, "Bu taramanın fotoğrafı bulunamadı (eski bir kayıt olabilir).")
    for old in USER_DATASET.glob(f"*/{id}.jpg"):  # etiketi değiştirildiyse eski kopyayı kaldır
        old.unlink()
    dest = USER_DATASET / body.label
    dest.mkdir(parents=True, exist_ok=True)
    dest.joinpath(f"{id}.jpg").write_bytes(src.read_bytes())
    await db.set_feedback(id, body.label)
    return _feedback_stats()


@router.get("/feedback/stats", response_model=FeedbackResult)
async def feedback_stats():
    return _feedback_stats()


def _feedback_stats() -> FeedbackResult:
    per = {d.name: len(list(d.glob("*.jpg"))) for d in USER_DATASET.iterdir() if d.is_dir()} if USER_DATASET.exists() else {}
    return FeedbackResult(total=sum(per.values()), per_class={k: v for k, v in per.items() if v})


async def _read_resistor(jpeg: bytes) -> ResistorReading:
    """Renk bantlarını klasik görüntü işleme (OpenCV) ile okur, ohm değerini Python hesaplar. Yapay zekâ kullanılmaz."""
    try:
        r = await asyncio.to_thread(read_bands, jpeg)
    except Exception as e:
        logger.warning("Renk bandı okuma hatası: %s", e)
        r = {"ok": False, "bands": [], "reason": ""}
    if r["ok"] and resistor.is_valid(r["bands"]):
        return ResistorReading(
            bands=r["bands"],
            detected=True,
            note="Renkler OpenCV görüntü işleme ile otomatik okundu; lütfen fotoğrafla karşılaştırıp kontrol et.",
            value=ResistorValue(**resistor.calculate(r["bands"])),
        )
    bands = resistor.DEFAULT_BANDS
    note = "Renk bantları fotoğraftan güvenilir şekilde okunamadı"
    note += f" ({r['reason'].rstrip('.')})" if r.get("reason") else ""
    note += ". Yanlış değer göstermemek için otomatik sonuç verilmedi; bantlara tıklayıp renkleri seçerek değeri hesapla."
    return ResistorReading(bands=bands, detected=False, note=note, value=None)


@router.get("/resistor/value", response_model=ResistorValue)
async def resistor_value(bands: str):
    """Ör: /api/resistor/value?bands=brown,black,red,gold -> 1 kΩ ±%5"""
    try:
        return resistor.calculate([b.strip().lower() for b in bands.split(",") if b.strip()])
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.post("/predict/live", response_model=LivePrediction)
async def predict_live(file: UploadFile = File(...)):
    """Canlı kamera için hızlı tahmin: sadece CNN, kayıt tutulmaz."""
    data = await file.read(MAX_BYTES + 1)
    if not data or len(data) > MAX_BYTES:
        raise HTTPException(400, "Geçersiz kare.")
    try:
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(415, "Geçerli bir görsel değil.")
    if not model_available():
        raise HTTPException(503, "CNN modeli henüz eğitilmedi.")
    classes = model_classes()
    probs = await asyncio.to_thread(predict_bytes, _normalized_jpeg(img), LIVE_VIEWS)
    order = sorted(range(len(classes)), key=lambda i: -probs[i])
    ranked = [ClassProb(key=classes[i], name=_display_name(classes[i]), probability=round(probs[i], 4)) for i in order]
    return LivePrediction(label=ranked[0].key, name=ranked[0].name, confidence=ranked[0].probability, top3=ranked[:3], probs=ranked)


@router.get("/predictions", response_model=list[Prediction])
async def list_predictions():
    return await db.latest_predictions(24)


@router.delete("/predictions/{id}")
async def delete_prediction(id: str):
    if not await db.delete_prediction(id):
        raise HTTPException(404, "Kayıt bulunamadı.")
    # eğitim setine eklenmiş kopya (dataset_user) kalır; sadece geçmişteki fotoğraf silinir
    (UPLOADS_DIR / f"{id}.jpg").unlink(missing_ok=True)
    return {"deleted": id}


@router.get("/model", response_model=ModelInfo)
async def model_info():
    path = ML_DIR / "weights" / "metrics.json"
    if not path.exists():
        raise HTTPException(503, "Model metrikleri henüz yok.")
    m = json.loads(path.read_text(encoding="utf-8"))
    for pc in m["per_class"]:
        pc["name"] = _display_name(pc["key"])
    return ModelInfo(
        **m,
        available=model_available(),
        class_names=[_display_name(k) for k in m["classes"]],
        ai_policy=AI_POLICY,
        ai_model=AI_MODEL,
    )


@router.get("/samples", response_model=list[Sample])
async def list_samples():
    if not SAMPLES_DIR.exists():
        return []
    out = []
    for p in sorted(SAMPLES_DIR.glob("*.jpg")):
        key = p.stem
        if key in COMPONENTS_BY_KEY:
            out.append(Sample(key=key, name=COMPONENTS_BY_KEY[key]["name"], url=f"/api/samples/{p.name}"))
    return out


@router.get("/samples/{filename}")
async def get_sample(filename: str):
    p = SAMPLES_DIR / Path(filename).name
    if not p.exists() or p.suffix != ".jpg":
        raise HTTPException(404, "Örnek bulunamadı.")
    return FileResponse(p, media_type="image/jpeg")
