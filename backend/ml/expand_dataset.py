"""
Veri setini GENİŞLETİR (mevcut görselleri silmeden): zayıf sınıflar için web'den yeni görseller indirir,
CLIP ile temizler, mevcut görsellerle neredeyse aynı olanları atar ve ml/dataset/<sinif>/ altına ekler.
Ayrıca "bilinmiyor" (bileşen değil: masa, el, klavye, kablo...) sınıfını oluşturur — canlı kamerada
kadrajda bileşen yokken modelin rastgele bir bileşen adı söylemesini engeller.

Yeni görseller "w2_" önekiyle kaydedilir; eğitimde doğrulama seti (weights/split.json) sabit tutulur,
böylece eski ve yeni model aynı görseller üzerinde adil karşılaştırılır.

Çalıştırma:  cd backend && python -m ml.expand_dataset            (tüm hedef sınıflar)
             cd backend && python -m ml.expand_dataset role,stm32 (sadece belirtilen sınıflar)
"""
from __future__ import annotations

import hashlib
import io
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from PIL import Image

from ml.classes import CLASSES, CLIP_NEGATIVE_PROMPTS, CLIP_PROMPTS, UNKNOWN

ROOT = Path(__file__).parent
RAW_DIR = ROOT / "raw_extra"
OUT_DIR = ROOT / "dataset"
IMG_SIZE = 256
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36"}

# Sınıf başına eklenecek en fazla yeni görsel
ADD_LIMIT = {UNKNOWN: 450, "sigorta": 260}
DEFAULT_ADD_LIMIT = 160

QUERIES: dict[str, list[str]] = {
    "role": ["5v relay pcb songle srd-05vdc", "blue sugar cube relay", "pcb relay isolated", "relay module 1 channel arduino",
             "omron pcb relay g5le", "small signal relay dip", "12v pcb relay black", "relay on breadboard", "hongfa relay pcb", "2 channel relay module"],
    "transformator": ["small pcb transformer 220v 12v", "ei core transformer small", "toroidal transformer copper", "mains transformer 9v",
                      "step down transformer close up", "laminated transformer isolated white", "audio transformer small", "center tap transformer 12-0-12",
                      "power transformer electronics hobby", "encapsulated pcb transformer"],
    "esp32": ["esp32 devkitc", "nodemcu esp32 board", "esp32 wroom 32 devkit v1 top", "esp32 board hand", "esp32 cam module",
              "esp32 s3 board", "esp32 on breadboard", "lolin d32 board", "esp32 devkit 38 pin top view", "esp32 c3 supermini"],
    "stm32": ["stm32 blue pill top view", "stm32f103c8t6 blue pill breadboard", "stm32f411 black pill weact", "stm32 nucleo f401re",
              "nucleo-144 board", "stm32 nucleo l476rg", "stm32 discovery board f407", "blue pill board in hand", "stm32f103 minimum system board", "black pill board breadboard"],
    "diyot": ["1n4007 diode macro", "1n4148 diode macro photo", "black diode silver stripe", "schottky diode 1n5819 axial", "diode on breadboard",
              "rectifier diode pile", "glass signal diode orange", "zener diode macro", "diode in hand", "6a10 diode"],
    "led": ["5mm led macro", "3mm led red", "led blue 5mm legs", "clear white led 5mm", "led on breadboard", "diffused led yellow",
            "led in hand", "rgb led common cathode", "assorted leds", "10mm led"],
    "transistor": ["bc547 macro", "2n2222 to-92", "to-220 transistor tip31", "mosfet irfz44n", "transistor on breadboard", "transistor in hand"],
    "elektrolitik_kondansator": ["electrolytic capacitor macro", "capacitor 10uf 50v", "capacitor on breadboard", "electrolytic capacitor in hand", "blue electrolytic capacitor", "large electrolytic capacitor 4700uf"],
    "kristal_osilator": ["hc-49 crystal macro", "crystal oscillator in hand", "16mhz crystal on breadboard", "crystal oscillator 4 pin can"],
    "buton": ["tactile switch macro", "push button on breadboard", "tactile button in hand", "big tactile button 12mm colored cap"],
    "potansiyometre": ["potentiometer on breadboard", "potentiometer in hand", "slide potentiometer", "10k potentiometer macro"],
    "sigorta": ["glass fuse in hand", "glass fuse macro", "pcb fuse holder fuse", "resettable fuse ptc",
                # otomatik sigorta (MCB): Türkçede o da "sigorta" denir
                "miniature circuit breaker single pole", "mcb b16 din rail", "otomatik sigorta", "mcb circuit breaker c32",
                "circuit breaker din rail white", "2 pole mcb breaker", "schneider mcb", "abb mcb s201", "mcb breaker isolated white background",
                "circuit breakers in electrical panel", "w otomat sigorta", "mcb breaker in hand"],
}

# 3. tur: kamera testinde %80 altında kalan sınıflar için gerçek ortam (elde, breadboard'da, masada) fotoğrafları
QUERIES_ROUND3: dict[str, list[str]] = {
    "role": ["relay on pcb board close up", "relay module top view photo", "srd-05vdc-sl-c relay", "jqc-3ff relay", "relay macro photography",
             "automotive relay 4 pin", "4 channel relay module", "omron relay photo", "relay in hand photo", "finder relay"],
    "transformator": ["small transformer on workbench", "transformer 220v to 12v in hand", "pcb mount transformer photo", "toroidal transformer photo",
                      "old radio transformer", "transformer with wire leads", "laminated core transformer close up photo", "power transformer diy electronics"],
    "kristal_osilator": ["crystal oscillator on pcb close up", "16mhz crystal photo", "quartz crystal in hand", "crystal oscillator 4 pin metal can",
                         "32.768khz crystal", "hc-49u crystal", "crystal oscillator dip metal"],
    "diyot": ["1n4007 diode in hand", "diode on breadboard close up", "black diode white stripe macro", "diodes on table", "1n5408 diode", "fr107 diode"],
    "potansiyometre": ["potentiometer knob on breadboard", "10k potentiometer in hand", "rotary potentiometer close up photo", "trimpot 3296",
                       "potentiometer with nut and washer", "dual potentiometer"],
    "entegre": ["dip ic chip on breadboard", "ic chip in hand", "ic chips on table", "lm358 chip photo", "74hc00 ic", "atmega328p chip close up", "ne555 on breadboard"],
    "sigorta": ["glass fuse on table", "car fuses in hand", "blade fuse close up", "fuse holder with glass fuse"],
    "elektrolitik_kondansator": ["capacitors on breadboard", "electrolytic capacitor in hand photo", "capacitor 1000uf photo", "electrolytic capacitors pile"],
}

QUERIES[UNKNOWN] = [
    "empty wooden desk top view", "white desk surface", "hand palm close up", "computer keyboard close up", "smartphone on table",
              "pen on desk", "coffee mug on desk", "usb cable on table", "headphones on desk", "wall texture plain", "webcam selfie person",
              "notebook paper desk", "screwdriver set", "empty breadboard", "digital multimeter", "screws and nuts", "keys on table",
              "computer mouse", "plant pot desk", "tv remote control", "soldering iron", "wire spool", "jumper wires bundle",
              "laptop on desk", "cardboard box", "fabric texture", "ceiling", "office chair", "water bottle", "eyeglasses on table"]


def _save(data: bytes, cls: str, tag: str) -> None:
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        return
    if min(img.size) < 96:
        return
    img.thumbnail((IMG_SIZE, IMG_SIZE))
    d = RAW_DIR / cls
    d.mkdir(parents=True, exist_ok=True)
    img.save(d / f"{tag}.jpg", quality=90)


def _download(job: tuple[str, str, str]) -> None:
    url, cls, tag = job
    if (RAW_DIR / cls / f"{tag}.jpg").exists():
        return
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code == 200 and len(r.content) > 3000:
            _save(r.content, cls, tag)
    except Exception:
        pass


def collect(classes: list[str]) -> None:
    from ddgs import DDGS

    jobs: list[tuple[str, str, str]] = []
    for cls in classes:
        for q in QUERIES.get(cls, []):
            res = []
            for attempt in range(3):
                try:
                    res = DDGS().images(q, max_results=80)
                    break
                except Exception as e:
                    print("arama hatası", q, e)
                    time.sleep(3 + attempt * 3)
            jobs += [(r["image"], cls, "w2_" + hashlib.md5(r["image"].encode()).hexdigest()[:12]) for r in res if r.get("image")]
            print(f"{cls:26s} {q:45s} -> {len(res)}", flush=True)
            time.sleep(1.0)
    with ThreadPoolExecutor(24) as ex:
        list(ex.map(_download, jobs))


def clean_and_add(classes: list[str]) -> None:
    import open_clip
    import torch

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k", device=dev)
    tok = open_clip.get_tokenizer("ViT-B-32")
    model.eval()
    comp = [c for c in CLASSES if c != UNKNOWN]
    texts = [CLIP_PROMPTS[c] for c in comp] + CLIP_NEGATIVE_PROMPTS
    with torch.no_grad():
        tf = model.encode_text(tok(texts).to(dev))
        tf = tf / tf.norm(dim=-1, keepdim=True)

    def embed(paths: list[Path]):
        out = []
        for i in range(0, len(paths), 64):
            ims, ok = [], []
            for p in paths[i : i + 64]:
                try:
                    ims.append(preprocess(Image.open(p).convert("RGB")))
                    ok.append(p)
                except Exception:
                    pass
            if not ims:
                continue
            with torch.no_grad():
                f = model.encode_image(torch.stack(ims).to(dev))
                f = f / f.norm(dim=-1, keepdim=True)
            out += list(zip(ok, f))
        return out

    # elle reddedilen görseller (ör. şema/trafo merkezi transformatör fotoğrafları) bir daha eklenmez
    bl_file = ROOT / "dataset_blacklist.txt"
    blacklist = set(bl_file.read_text(encoding="utf-8-sig").split()) if bl_file.exists() else set()
    for cls in classes:
        raw = sorted((RAW_DIR / cls).glob("*.jpg")) if (RAW_DIR / cls).exists() else []
        new = embed([p for p in raw if f"{cls}/{p.name}" not in blacklist])
        dest = OUT_DIR / cls
        dest.mkdir(parents=True, exist_ok=True)
        existing = [e for _, e in embed(sorted(dest.glob("*.jpg")))]
        # bu sınıfın mevcut görselleri; yeni eklenenler de buna katılır (tekrarları önlemek için)
        bank = torch.stack(existing) if existing else torch.empty(0, tf.shape[1], device=dev)
        scored = []
        for p, e in new:
            probs = (100.0 * e @ tf.T).softmax(-1)
            top = probs.topk(2).indices.tolist()
            if cls == UNKNOWN:
                # bileşen değil: hiçbir bileşen metnine belirgin benzemiyor olmalı
                if float(probs[: len(comp)].max()) < 0.30:
                    scored.append((1.0 - float(probs[: len(comp)].max()), p, e))
            else:
                ci = comp.index(cls)
                if top[0] == ci or (top[1] == ci and probs[ci] > 0.25 and top[0] < len(comp)):
                    scored.append((float(probs[ci]), p, e))
        scored.sort(key=lambda x: -x[0])
        limit = ADD_LIMIT.get(cls, DEFAULT_ADD_LIMIT)
        added = 0
        for _, p, e in scored:
            if added >= limit:
                break
            if bank.shape[0] and float((bank @ e).max()) > 0.93:  # neredeyse aynı görsel zaten var
                continue
            if (dest / p.name).exists():
                continue
            Image.open(p).convert("RGB").save(dest / p.name, quality=90)
            bank = torch.cat([bank, e[None]])
            added += 1
        print(f"{cls:26s} indirilen={len(new):4d} CLIP'ten geçen={len(scored):4d} eklenen={added:4d} toplam={len(list(dest.glob('*.jpg')))}", flush=True)


# Kamera benzetimi için arka plan havuzu (sınıf değil; train.py bileşenleri bunların üzerine yapıştırır)
BG_DIR = ROOT / "backgrounds"
BACKGROUND_QUERIES = [
    "wooden table texture top view", "white desk surface top view", "black desk surface", "gray desk mat top view",
    "green cutting mat top view", "self healing cutting mat", "empty breadboard top view", "white paper sheet on table",
    "open palm hand top view", "hand palm close up", "laboratory bench surface", "workbench top view",
    "office desk top view", "granite countertop texture", "carpet texture", "bed sheet texture",
    "plastic storage box top view empty", "anti static mat", "electronics workbench messy top view", "notebook on desk top view",
    "laptop keyboard close up", "concrete floor texture", "kitchen table top view", "school desk surface",
]


def collect_backgrounds() -> None:
    from ddgs import DDGS

    jobs = []
    for q in BACKGROUND_QUERIES:
        try:
            res = DDGS().images(q, max_results=60)
        except Exception as e:
            print("arama hatası", q, e)
            continue
        jobs += [(r["image"], "_bg", "bg_" + hashlib.md5(r["image"].encode()).hexdigest()[:12]) for r in res if r.get("image")]
        print(f"arka plan {q:45s} -> {len(res)}", flush=True)
        time.sleep(1.0)
    with ThreadPoolExecutor(24) as ex:
        list(ex.map(_download, jobs))
    BG_DIR.mkdir(exist_ok=True)
    for p in (RAW_DIR / "_bg").glob("*.jpg"):
        p.replace(BG_DIR / p.name)
    print("arka plan sayısı:", len(list(BG_DIR.glob("*.jpg"))))


# Yeni eklenen (w2_) görsellerden atılacak türler: çizimler, metinli tanıtım görselleri, şebeke ekipmanı
PRUNE_PROMPTS = [
    "a schematic diagram or technical drawing",
    "an infographic or advertisement with a lot of text",
    "a pinout diagram with colored labels",
    "a large utility power transformer outdoors on a pole or in a substation",
    "an electrical substation or power lines",
    "a cartoon or 3d illustration",
]


def prune_new(classes: list[str]) -> None:
    """w2_ görsellerinde kendi sınıf metninden çok çizim/infografik/şebeke ekipmanına benzeyenleri siler."""
    import open_clip
    import torch

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k", device=dev)
    tok = open_clip.get_tokenizer("ViT-B-32")
    model.eval()
    for cls in classes:
        if cls == UNKNOWN:
            continue
        with torch.no_grad():
            tf = model.encode_text(tok([CLIP_PROMPTS[cls]] + PRUNE_PROMPTS).to(dev))
            tf = tf / tf.norm(dim=-1, keepdim=True)
        files = sorted((OUT_DIR / cls).glob("w2_*.jpg"))
        removed = 0
        for i in range(0, len(files), 64):
            batch = files[i : i + 64]
            with torch.no_grad():
                f = model.encode_image(torch.stack([preprocess(Image.open(p).convert("RGB")) for p in batch]).to(dev))
                f = f / f.norm(dim=-1, keepdim=True)
                probs = (100.0 * f @ tf.T).softmax(-1)
            for p, pr in zip(batch, probs):
                if int(pr.argmax()) != 0:
                    p.unlink()
                    removed += 1
        print(f"{cls:26s} silinen={removed:4d} kalan={len(list((OUT_DIR / cls).glob('*.jpg')))}", flush=True)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    targets = args[0].split(",") if args else list(QUERIES)
    if "--backgrounds" in sys.argv:
        collect_backgrounds()
        sys.exit(0)
    if "--round3" in sys.argv:
        QUERIES.clear()
        QUERIES.update(QUERIES_ROUND3)
        targets = args[0].split(",") if args else list(QUERIES_ROUND3)
    if "--prune-only" in sys.argv:
        prune_new(targets)
        sys.exit(0)
    if "--clean-only" not in sys.argv:
        collect(targets)
    clean_and_add(targets)
    prune_new(targets)
