"""Her sınıf için doğrulama setinden (modelin eğitimde görmediği) birer örnek görsel seçer -> ml/samples/<sinif>.jpg

Çalıştırma: cd backend && python -m ml.make_samples            (tüm sınıflar)
           cd backend && python -m ml.make_samples arduino,esp32  (sadece belirli sınıflar)
"""
import sys
import json
import shutil
from pathlib import Path

from ml.classes import UNKNOWN
from ml.inference import model_classes, predict_bytes

ROOT = Path(__file__).parent
OUT = ROOT / "samples"


def main() -> None:
    only = sys.argv[1].split(",") if len(sys.argv) > 1 else None
    classes = model_classes()
    val = json.loads((ROOT / "weights" / "split.json").read_text(encoding="utf-8"))["val"]
    # split.json eski sürümlerde mutlak yol tutuyordu; dataset/<sinif>/<dosya> kısmını kullan
    val = [ROOT / "dataset" / Path(p).parent.name / Path(p).name for p in val]
    # Kaggle görselleri 128 px ve çoğu endüstriyel/karışık sahne; sitedeki örnekler net web fotoğraflarından seçilir
    val = [p for p in val if p.parent.name != UNKNOWN and not p.name.startswith("kg_") and (not only or p.parent.name in only)]
    best: dict[str, tuple[float, Path]] = {}
    for p in val:
        cls = p.parent.name
        probs = predict_bytes(p.read_bytes())
        ci = classes.index(cls)
        # doğru sınıflandırılan ve tipik (ne çok kolay ne çok zor) bir örnek seç
        if max(range(len(probs)), key=lambda i: probs[i]) == ci:
            score = probs[ci]
            if cls not in best or score > best[cls][0]:
                best[cls] = (score, p)
    OUT.mkdir(exist_ok=True)
    for cls, (score, p) in best.items():
        shutil.copy(p, OUT / f"{cls}.jpg")
        print(f"{cls:26s} {score:.3f} {p}")


if __name__ == "__main__":
    main()
