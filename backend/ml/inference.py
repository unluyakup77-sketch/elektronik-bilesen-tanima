"""Çıkarım (inference): eğitilmiş CNN'i yükler ve bir görsel için sınıf olasılıklarını döndürür."""
from __future__ import annotations

import io
import json
import threading
from pathlib import Path

import torch
from PIL import Image, ImageOps, ImageStat
from torchvision import transforms

from ml.train import MEAN, STD, build_model

WEIGHTS_DIR = Path(__file__).parent / "weights"
WEIGHTS = WEIGHTS_DIR / "model.pt"
# Test-time augmentation görünüm sayısı (bkz. views_of)
UPLOAD_VIEWS = 6
LIVE_VIEWS = 4  # canlı kareler zaten kare kırpılır (kırpmasız görünüm gereksiz); ~2 kat hızlı
_model: torch.nn.Module | None = None
_size = 224
_lock = threading.Lock()


def _metrics() -> dict:
    return json.loads((WEIGHTS_DIR / "metrics.json").read_text(encoding="utf-8"))


def model_classes() -> list[str]:
    """Modelin eğitildiği sınıf listesi (metrics.json'dan) — çıkış katmanının sırası budur."""
    return _metrics()["classes"]


def get_model() -> torch.nn.Module:
    global _model, _size
    with _lock:
        if _model is None:
            m = _metrics()
            # eski metrics.json dosyalarında "arch" yoktur: o modeller EfficientNet-B0'dır
            net = build_model(len(m["classes"]), pretrained=False, arch=m.get("arch", "efficientnet_b0"))
            state = torch.load(WEIGHTS, map_location="cpu")
            net.load_state_dict({k: v.float() for k, v in state.items()})  # yarı hassasiyetle kaydedilmiş olabilir
            net.eval()
            _size = m.get("input_size", 224)
            _model = net
    return _model


def model_available() -> bool:
    return WEIGHTS.exists() and (WEIGHTS_DIR / "metrics.json").exists()


_to_tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def _center(img: Image.Image, size: int, zoom: float = 1.0) -> Image.Image:
    """Kısa kenarın `zoom` oranındaki merkez kare (zoom<1 yakınlaştırır: kadrajda küçük kalan parça için)."""
    w, h = img.size
    s = int(min(w, h) * zoom)
    return img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2)).resize((size, size), Image.BILINEAR)


def _letterbox(img: Image.Image, size: int) -> Image.Image:
    """Görüntünün tamamı (kırpmadan), kenarlar ortalama renkle doldurulur: kenardaki parçalar kaybolmaz."""
    fill = tuple(int(v) for v in ImageStat.Stat(img).mean[:3])
    return ImageOps.pad(img, (size, size), color=fill, method=Image.BILINEAR)


def views_of(img: Image.Image, size: int, n: int) -> list[torch.Tensor]:
    """n=2: merkez · n=4: + yakınlaştırılmış merkez · n=6: + kırpmasız tüm görüntü (her biri + ayna görüntüsü)."""
    base = [_center(img, size)]
    if n >= 4:
        base.append(_center(img, size, zoom=0.7))
    if n >= 6:
        base.append(_letterbox(img, size))
    out = []
    for v in base:
        t = _to_tensor(v)
        out += [t, torch.flip(t, dims=[2])]
    return out


def tta_batch(net: torch.nn.Module, imgs: list[Image.Image], size: int, n: int, dev: str = "cpu") -> torch.Tensor:
    """Birden çok görselin, her biri için n görünümün ortalama softmax olasılıkları: [len(imgs), sınıf]."""
    x = torch.stack([t for im in imgs for t in views_of(im.convert("RGB"), size, n)]).to(dev)
    k = len(x) // len(imgs)
    return net(x).float().softmax(1).view(len(imgs), k, -1).mean(1)


def predict_bytes(data: bytes, views: int = UPLOAD_VIEWS) -> list[float]:
    """Görseli sınıflandırır (çok görünümlü test-time augmentation)."""
    model = get_model()
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    with torch.no_grad():
        return tta_batch(model, [img], _size, views)[0].tolist()
