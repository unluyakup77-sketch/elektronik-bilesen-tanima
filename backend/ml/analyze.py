"""
Hata analizi: eğitilmiş modeli doğrulama setinde çalıştırır, yanlış sınıflandırılan görselleri listeler
ve her karışıklık çifti için bir kolaj (ml/analysis/<gercek>__<tahmin>.jpg) üretir.

Çalıştırma:  cd backend && python -m ml.analyze
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from ml import inference
from ml.inference import get_model, model_classes

ROOT = Path(__file__).parent
OUT = ROOT / "analysis"


def main() -> None:
    # isteğe bağlı: başka bir eğitim çıktısını incele  (python -m ml.analyze ml/runs/convnext_tiny)
    if len(sys.argv) > 1:
        inference.WEIGHTS_DIR = Path(sys.argv[1]).resolve()
        inference.WEIGHTS = inference.WEIGHTS_DIR / "model.pt"
    classes = model_classes()
    val = json.loads((inference.WEIGHTS_DIR / "split.json").read_text(encoding="utf-8"))["val"]
    model = get_model()
    wrong: dict[tuple[str, str], list[tuple[str, float]]] = defaultdict(list)
    correct = 0
    for i in range(0, len(val), 64):
        paths = [Path(p) for p in val[i : i + 64]]
        # split.json eğitim ortamındaki mutlak yolları içerir; dataset/<sinif>/<dosya> kısmını kullan
        paths = [ROOT / "dataset" / p.parent.name / p.name for p in paths]
        with torch.no_grad():
            probs = inference.tta_batch(model, [Image.open(p) for p in paths], inference._size, inference.UPLOAD_VIEWS)
        conf, pred = probs.max(1)
        for p, c, pr in zip(paths, conf.tolist(), pred.tolist()):
            true = p.parent.name
            if classes[pr] == true:
                correct += 1
            else:
                wrong[(true, classes[pr])].append((str(p), c))
    total = len(val)
    print(f"doğruluk: {correct}/{total} = {correct / total:.4f}")
    pairs = sorted(wrong.items(), key=lambda kv: -len(kv[1]))
    print("\nEn sık karışıklıklar (gerçek -> tahmin):")
    for (t, p), items in pairs[:15]:
        print(f"  {t:26s} -> {p:26s} {len(items)}")

    OUT.mkdir(exist_ok=True)
    for f in OUT.glob("*.jpg"):
        f.unlink()
    for (t, p), items in pairs:
        if len(items) < 2:
            continue
        tiles = items[:12]
        W = 160
        sheet = Image.new("RGB", (W * min(6, len(tiles)), W * ((len(tiles) + 5) // 6)), "white")
        for k, (path, c) in enumerate(tiles):
            im = Image.open(path).convert("RGB")
            im.thumbnail((W, W))
            sheet.paste(im, ((k % 6) * W, (k // 6) * W))
            ImageDraw.Draw(sheet).text(((k % 6) * W + 3, (k // 6) * W + 3), f"{Path(path).name[:14]} {c:.2f}", fill="red")
        sheet.save(OUT / f"{t}__{p}.jpg", quality=85)
    (OUT / "wrong.json").write_text(json.dumps({f"{t}->{p}": items for (t, p), items in pairs}, indent=1))
    print("\nper-class doğrulama sayısı:", dict(Counter(Path(p).parent.name for p in val)))


if __name__ == "__main__":
    main()
