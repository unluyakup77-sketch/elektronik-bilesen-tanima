"""
Veri seti hazırlama scripti.

Adımlar:
 1. Kaggle "aryaminus/electronic-components" ve HuggingFace "qipchip31/electronic_components"
    veri setlerinden uygun klasörleri alır (varsa).
 2. Her sınıf için DuckDuckGo görsel aramasıyla ek görseller indirir.
 3. CLIP (ViT-B-32) ile otomatik temizlik yapar: görselin en çok benzediği metin kendi sınıfı
    değilse (ör. dolu bir devre kartı, şema çizimi, alakasız fotoğraf) atılır.
 4. Neredeyse aynı (duplicate) görselleri siler, 224px'e küçültüp ml/dataset/<sinif>/ altına kaydeder.

Çalıştırma:  cd backend && python -m ml.build_dataset
"""
from __future__ import annotations

import hashlib
import io
import os
import random
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from PIL import Image

import sys

from ml.classes import CLASSES, CLIP_NEGATIVE_PROMPTS, CLIP_PROMPTS, EXTRA_QUERIES, SEARCH_QUERIES

ROOT = Path(__file__).parent
RAW_DIR = ROOT / "raw"
OUT_DIR = ROOT / "dataset"
KAGGLE_DIR = Path(os.environ.get("KAGGLE_DIR", "/app/ml_src/kag/images"))
HF_PARQUET = [Path("/app/ml_src/train.parquet"), Path("/app/ml_src/test.parquet")]
MAX_PER_CLASS = 420
IMG_SIZE = 256

KAGGLE_MAP = {
    "led": ["LED"],
    "seramik_kondansator": ["Bypass-capacitor"],
    "elektrolitik_kondansator": ["Electrolytic-capacitor"],
    "transistor": ["transistor", "junction-transistor", "PNP-transistor"],
    "diyot": ["semiconductor-diode"],
    "entegre": ["Integrated-micro-circuit", "microchip"],
    "potansiyometre": ["potentiometer"],
    "bobin": ["induction-coil"],
    "role": ["relay", "electric-relay"],
    "sigorta": ["cartridge-fuse"],
    "transformator": ["step-down-transformer", "step-up-transformer"],
}
HF_MAP = {0: "direnc", 2: "bobin", 3: "transistor"}  # 1=capacitor (tip belirsiz, alınmaz)

HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120 Safari/537.36"}


def save_image(img: Image.Image, cls: str, tag: str) -> None:
    img = img.convert("RGB")
    if min(img.size) < 64:
        return
    img.thumbnail((IMG_SIZE, IMG_SIZE))
    d = RAW_DIR / cls
    d.mkdir(parents=True, exist_ok=True)
    img.save(d / f"{tag}.jpg", quality=88)


def collect_local() -> None:
    for cls, folders in KAGGLE_MAP.items():
        for f in folders:
            src = KAGGLE_DIR / f
            if not src.exists():
                continue
            for p in src.iterdir():
                try:
                    save_image(Image.open(p), cls, f"kg_{hashlib.md5(p.name.encode()).hexdigest()[:10]}")
                except Exception:
                    pass
    try:
        import numpy as np
        import pandas as pd

        for pq in HF_PARQUET:
            if not pq.exists():
                continue
            df = pd.read_parquet(pq)
            for i, row in df.iterrows():
                cls = HF_MAP.get(int(row["label"]))
                if not cls:
                    continue
                arr = np.stack([np.stack(r) for r in row["image"]]).astype("uint8")
                save_image(Image.fromarray(arr), cls, f"hf_{pq.stem}_{i}")
    except Exception as e:  # pragma: no cover
        print("HF parquet okunamadı:", e)


def _download(args: tuple[str, str, str]) -> None:
    url, cls, tag = args
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code != 200 or len(r.content) < 2000:
            return
        save_image(Image.open(io.BytesIO(r.content)), cls, tag)
    except Exception:
        pass


def collect_web(queries: dict[str, list[str]] = SEARCH_QUERIES) -> None:
    from ddgs import DDGS

    jobs: list[tuple[str, str, str]] = []
    for cls in CLASSES:
        for q in queries.get(cls, []):
            for attempt in range(3):
                try:
                    res = DDGS().images(q, max_results=120)
                    break
                except Exception as e:
                    print("arama hatası", q, e)
                    time.sleep(3 + attempt * 3)
                    res = []
            for r in res:
                url = r.get("image")
                if url:
                    jobs.append((url, cls, "web_" + hashlib.md5(url.encode()).hexdigest()[:12]))
            print(f"{cls:26s} {q:45s} -> {len(res)}")
            time.sleep(1.0)
    with ThreadPoolExecutor(32) as ex:
        list(ex.map(_download, jobs))


def clean_with_clip(only: list[str] | None = None) -> None:
    import open_clip
    import torch

    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
    tok = open_clip.get_tokenizer("ViT-B-32")
    model.eval()
    # "bilinmiyor" sınıfı burada değil, ml/expand_dataset.py ile oluşturulur
    texts = [CLIP_PROMPTS[c] for c in CLASSES if c in CLIP_PROMPTS] + CLIP_NEGATIVE_PROMPTS
    with torch.no_grad():
        tf = model.encode_text(tok(texts))
        tf = tf / tf.norm(dim=-1, keepdim=True)

    if only is None and OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    for ci, cls in enumerate(CLASSES):
        if cls not in CLIP_PROMPTS or (only is not None and cls not in only):
            continue
        files = sorted((RAW_DIR / cls).glob("*.jpg")) if (RAW_DIR / cls).exists() else []
        kept: list[tuple[float, Path, torch.Tensor]] = []
        for i in range(0, len(files), 64):
            batch = files[i : i + 64]
            ims = []
            for p in batch:
                try:
                    ims.append(preprocess(Image.open(p).convert("RGB")))
                except Exception:
                    ims.append(None)
            valid = [(p, t) for p, t in zip(batch, ims) if t is not None]
            if not valid:
                continue
            with torch.no_grad():
                f = model.encode_image(torch.stack([t for _, t in valid]))
                f = f / f.norm(dim=-1, keepdim=True)
                probs = (100.0 * f @ tf.T).softmax(dim=-1)
            for (p, _), pr, emb in zip(valid, probs, f):
                top2 = pr.topk(2).indices.tolist()
                # kendi sınıfı 1. sırada olmalı (veya 2. sırada ve yeterince yüksek olasılıkta)
                if top2[0] == ci or (top2[1] == ci and pr[ci] > 0.25 and top2[0] < len(CLASSES)):
                    kept.append((float(pr[ci]), p, emb))
        kept.sort(key=lambda x: -x[0])
        final: list[tuple[Path, torch.Tensor]] = []
        for score, p, emb in kept:
            if any(float(emb @ e) > 0.95 for _, e in final):  # duplicate
                continue
            final.append((p, emb))
            if len(final) >= MAX_PER_CLASS:
                break
        d = OUT_DIR / cls
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
        for p, _ in final:
            shutil.copy(p, d / p.name)
        print(f"{cls:26s} ham={len(files):4d}  temiz={len(final):4d}")


if __name__ == "__main__":
    random.seed(42)
    # Sadece belirli sınıfları ekle/güncelle:  python -m ml.build_dataset --only arduino,esp32,stm32
    if "--only" in sys.argv:
        only = sys.argv[sys.argv.index("--only") + 1].split(",")
        collect_web({c: SEARCH_QUERIES[c] for c in only})
        clean_with_clip(only)
        sys.exit(0)
    if not RAW_DIR.exists():
        collect_local()
        collect_web()
    if "--more" in sys.argv:
        collect_web(EXTRA_QUERIES)
    clean_with_clip()
