"""
Kamera koşullarında belirli sınıfların hangi görsellerde hata yaptığını kaynağa göre (kg_ = Kaggle, web_/w2_ = web)
ayırır ve hatalı görsellerden kolaj üretir: ml/analysis/fail_<sinif>.jpg

Çalıştırma: cd backend && python -m ml.failures ml/runs/convnext_v4 role,transformator,sigorta
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

import torch
from PIL import Image

from ml.inference import tta_batch
from ml.stress_test import MODES, make
from ml.train import build_model

ROOT = Path(__file__).parent


def main() -> None:
    run, targets = Path(sys.argv[1]), sys.argv[2].split(",")
    m = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    net = build_model(len(m["classes"]), pretrained=False, arch=m["arch"])
    net.load_state_dict({k: v.float() for k, v in torch.load(run / "model.pt", map_location="cpu").items()})
    net.eval()
    val = [ROOT / "dataset" / Path(p).parent.name / Path(p).name for p in json.loads((ROOT / "weights" / "split.json").read_text(encoding="utf-8"))["val"]]
    bgs = [p for p in val if p.parent.name == "bilinmiyor"]
    (ROOT / "analysis").mkdir(exist_ok=True)
    for cls in targets:
        items = [p for p in val if p.parent.name == cls]
        fails = Counter()
        totals = Counter()
        bad: dict[Path, int] = Counter()
        for kind, scale in MODES[1:]:
            rng = random.Random(0)
            imgs = [make(p, kind, scale, bgs, rng) for p in items]
            with torch.no_grad():
                pr = tta_batch(net, imgs, m.get("input_size", 224), 2)
            for p, k in zip(items, pr.argmax(1).tolist()):
                src = p.name.split("_")[0]
                totals[src] += 1
                if m["classes"][k] != cls:
                    fails[src] += 1
                    bad[p] += 1
        print(cls, {s: f"{fails[s]}/{totals[s]} hata (%{fails[s] / totals[s] * 100:.0f})" for s in totals})
        worst = [p for p, _ in bad.most_common(24)]
        W = 120
        sheet = Image.new("RGB", (W * 8, W * ((len(worst) + 7) // 8 or 1)), "white")
        for i, p in enumerate(worst):
            im = Image.open(p).convert("RGB")
            im.thumbnail((W, W))
            sheet.paste(im, ((i % 8) * W, (i // 8) * W))
        sheet.save(ROOT / "analysis" / f"fail_{cls}.jpg")


if __name__ == "__main__":
    main()
