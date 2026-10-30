"""
Gerçek kamera koşullarını taklit eden zorlu test. Doğrulama setindeki (eğitimde görülmemiş) bileşen görselleri,
yine doğrulamadaki "bilinmiyor" görsellerinin (masa, el, kumaş...) üzerine küçültülerek yapıştırılır,
bulanıklık / JPEG / ışık bozulmaları eklenir. Sonuç: koşul bazında ve SINIF BAZINDA doğruluk.

Koşullar:
  orijinal            - fotoğraf olduğu gibi
  dikdörtgen <ölçek>  - ürün fotoğrafı arka planıyla birlikte masaya konur (internetten alınmış foto gibi)
  kesik <ölçek>       - parçanın arka planı silinip masaya konur (gerçek kamera görüntüsüne en yakın)

Çalıştırma:  cd backend && python -m ml.stress_test [model_klasoru] [--views N]
"""
from __future__ import annotations

import io
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
from PIL import Image, ImageEnhance, ImageFilter

from ml import camera_sim
from ml.classes import UNKNOWN
from ml.inference import tta_batch
from ml.train import build_model

ROOT = Path(__file__).parent
TARGET = 0.80


def degrade(img: Image.Image, rng: random.Random) -> Image.Image:
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.6, 1.3))
    img = ImageEnhance.Color(img).enhance(rng.uniform(0.6, 1.2))
    if rng.random() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.5, 1.5)))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=rng.randint(40, 80))
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


def rect_composite(fg: Image.Image, bg: Image.Image, scale: float, rng: random.Random) -> Image.Image:
    W = 480
    canvas = bg.convert("RGB").resize((W, W))
    fg = fg.copy()
    fg.thumbnail((int(W * scale), int(W * scale)))
    x = (W - fg.width) // 2 + rng.randint(-W // 12, W // 12)
    y = (W - fg.height) // 2 + rng.randint(-W // 12, W // 12)
    canvas.paste(fg, (max(0, x), max(0, y)))
    return canvas


class Ensemble(torch.nn.Module):
    """Modellerin logit'lerini değil, olasılıklarını ortalar; tta_batch softmax uyguladığı için log-olasılık döndürür."""

    def __init__(self, nets):
        super().__init__()
        self.nets = torch.nn.ModuleList(nets)

    def forward(self, x):
        return torch.stack([n(x).float().softmax(1) for n in self.nets]).mean(0).clamp_min(1e-8).log()


MODES = [("orijinal", None), ("dikdörtgen", 0.6), ("dikdörtgen", 0.4), ("kesik", 0.6), ("kesik", 0.4)]


def make(p: Path, kind: str, scale: float | None, bgs: list[Path], rng: random.Random) -> Image.Image:
    img = Image.open(p).convert("RGB")
    if kind == "dikdörtgen":
        return degrade(rect_composite(img, Image.open(rng.choice(bgs)), scale, rng), rng)
    if kind == "kesik":
        return camera_sim.degrade(camera_sim.composite(img, Image.open(rng.choice(bgs)), rng, scale=(scale, scale)), rng)
    return img


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    views = int(sys.argv[sys.argv.index("--views") + 1]) if "--views" in sys.argv else 2
    # birden çok klasör virgülle verilirse modellerin olasılıkları ortalanır (topluluk / ensemble)
    runs = [Path(r) for r in args[0].split(",")] if args else [ROOT / "weights"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    nets = []
    for run in runs:
        m = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
        n = build_model(len(m["classes"]), pretrained=False, arch=m.get("arch", "efficientnet_b0"))
        n.load_state_dict({k: v.float() for k, v in torch.load(run / "model.pt", map_location="cpu").items()})
        nets.append(n.to(dev).eval())
    classes = m["classes"]
    size = m.get("input_size", 224)
    net = nets[0] if len(nets) == 1 else Ensemble(nets)
    run = ",".join(str(r) for r in runs)

    val = [ROOT / "dataset" / Path(p).parent.name / Path(p).name for p in json.loads((ROOT / "weights" / "split.json").read_text(encoding="utf-8"))["val"]]
    comps = [p for p in val if p.parent.name != UNKNOWN]
    bgs = [p for p in val if p.parent.name == UNKNOWN]

    print(f"model: {run}  ({m.get('arch')})  görünüm(TTA)={views}  test görseli={len(comps)}")
    per_class: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # kamera koşulları (orijinal hariç)
    per_class_web: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # aynısı, Kaggle'ın endüstriyel görselleri hariç
    for kind, scale in MODES:
        rng = random.Random(0)
        correct = unk = 0
        wrong = Counter()
        for i in range(0, len(comps), 32):
            chunk = comps[i : i + 32]
            imgs = [make(p, kind, scale, bgs, rng) for p in chunk]
            with torch.no_grad():
                pr = tta_batch(net, imgs, size, views, dev)
            for p, k in zip(chunk, pr.argmax(1).tolist()):
                ok = classes[k] == p.parent.name
                correct += ok
                unk += classes[k] == UNKNOWN
                if not ok:
                    wrong[f"{p.parent.name}->{classes[k]}"] += 1
                if kind != "orijinal":
                    per_class[p.parent.name][0] += ok
                    per_class[p.parent.name][1] += 1
                    if not p.name.startswith("kg_"):  # Kaggle dışı (çoğunlukla hobi/eğitim tipi parça fotoğrafları)
                        per_class_web[p.parent.name][0] += ok
                        per_class_web[p.parent.name][1] += 1
        label = kind if scale is None else f"{kind} {scale}"
        print(f"  {label:16s} doğruluk={correct / len(comps):.3f}  'bilinmiyor'={unk / len(comps):.3f}  en sık hata: {wrong.most_common(3)}", flush=True)

    print(f"\n  Sınıf bazında (tüm kamera koşulları, hedef ≥ %{TARGET * 100:.0f}):        tümü      Kaggle hariç")
    worst = sorted(per_class.items(), key=lambda kv: kv[1][0] / kv[1][1])
    for cls, (ok, n) in worst:
        acc = ok / n
        wok, wn = per_class_web.get(cls, [0, 0])
        web = f"%{wok / wn * 100:5.1f}" if wn else "   -  "
        print(f"    {'✗' if acc < TARGET else '✓'} {cls:26s} %{acc * 100:5.1f}  ({ok}/{n})   {web}")
    total_ok = sum(v[0] for v in per_class.values())
    total_n = sum(v[1] for v in per_class.values())
    print(f"  Kamera koşulları ortalaması: %{total_ok / total_n * 100:.1f}  hedef altı sınıf: {sum(v[0] / v[1] < TARGET for v in per_class.values())}")


if __name__ == "__main__":
    main()
