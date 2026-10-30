"""
Eğitim çıktılarını AYNI doğrulama görselleri üzerinde karşılaştırır.

  - "Bileşen doğruluğu": eski modelin doğrulama setindeki 19 sınıflı bileşen görselleri (adil karşılaştırma)
  - "Yanlış ret": bileşen görseline "bilinmiyor" deme oranı
  - "Bilinmiyor doğruluğu": bileşen olmayan görsellerde "bilinmiyor" deme oranı (sadece yeni modellerde)

Çalıştırma:  cd backend && python -m ml.compare ml/weights ml/runs/efficientnet_b0 ml/runs/convnext_tiny
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image

from ml.classes import UNKNOWN
from ml.train import build_model, eval_transform

ROOT = Path(__file__).parent


def load(run: Path):
    m = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    net = build_model(len(m["classes"]), pretrained=False, arch=m.get("arch", "efficientnet_b0"))
    net.load_state_dict(torch.load(run / "model.pt", map_location="cpu"))
    return net.eval(), m["classes"], eval_transform(m.get("input_size", 224)), m


def val_files(run: Path) -> list[Path]:
    return [ROOT / "dataset" / Path(p).parent.name / Path(p).name for p in json.loads((run / "split.json").read_text(encoding="utf-8"))["val"]]


def main() -> None:
    runs = [Path(a) for a in sys.argv[1:]]
    base = {p for p in val_files(runs[0]) if p.parent.name != UNKNOWN}
    unknown_val = sorted({p for r in runs for p in val_files(r) if p.parent.name == UNKNOWN})
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"bileşen doğrulama görseli: {len(base)}  bilinmiyor doğrulama görseli: {len(unknown_val)}\n")
    for run in runs:
        net, classes, tf, m = load(run)
        net = net.to(dev)
        files = sorted(base) + (unknown_val if UNKNOWN in classes else [])
        correct = rejected = n = unk_ok = 0
        for i in range(0, len(files), 64):
            batch = files[i : i + 64]
            x = torch.stack([tf(Image.open(p).convert("RGB")) for p in batch]).to(dev)
            with torch.no_grad():
                pr = (net(x).softmax(1) + net(torch.flip(x, dims=[3])).softmax(1)) / 2
            for p, k in zip(batch, pr.argmax(1).tolist()):
                if p.parent.name == UNKNOWN:
                    unk_ok += classes[k] == UNKNOWN
                else:
                    n += 1
                    correct += classes[k] == p.parent.name
                    rejected += classes[k] == UNKNOWN
        # CPU'da tek görsel hızı (sunucu CPU'da çalışır)
        cpu = net.to("cpu")
        x = tf(Image.open(files[0]).convert("RGB"))[None].repeat(2, 1, 1, 1)
        with torch.no_grad():
            cpu(x)
            t0 = time.perf_counter()
            for _ in range(5):
                cpu(x)
        ms = (time.perf_counter() - t0) / 5 * 1000
        line = f"{run.as_posix():28s} {m.get('arch', 'efficientnet_b0'):18s} bileşen doğruluğu={correct / n:.4f}  yanlış ret={rejected / n:.4f}"
        if UNKNOWN in classes and unknown_val:
            line += f"  bilinmiyor doğruluğu={unk_ok / len(unknown_val):.4f}"
        print(line + f"  CPU={ms:.0f} ms/görsel", flush=True)


if __name__ == "__main__":
    main()
