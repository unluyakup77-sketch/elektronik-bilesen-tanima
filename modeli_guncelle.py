"""Sitede "Sonuç doğru mu?" ile kaydedilen fotoğraflarla modeli ince ayar yapar (NVIDIA GPU, ~10 dk),
yeni modeli eskisiyle karşılaştırır ve onay verilirse siteye yükler (önceki model ml/weights_onceki'ye yedeklenir).

Çalıştırma:  py modeli_guncelle.py
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent / "backend"
TRAIN_PY = BACKEND / ".venv-train" / "Scripts" / "python.exe"
ML = BACKEND / "ml"
FILES = ["model.pt", "metrics.json", "split.json"]


def run(args: list[str]) -> None:
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    if subprocess.call([str(TRAIN_PY), *args], cwd=BACKEND, env=env) != 0:
        print("Hata oluştu, yukarıdaki mesajı kontrol edin.")
        sys.exit(1)


def main() -> None:
    if not TRAIN_PY.exists():
        print('GPU eğitim ortamı yok. Kurulum için backend/ml/README.md - "Yeniden eğitim" bölümüne bakın.')
        sys.exit(1)

    print("[1/3] İnce ayar başlıyor (mevcut model + kaydettiğiniz fotoğraflar)...")
    run(["-u", "-m", "ml.train", "--arch", "convnext_tiny", "--epochs", "8", "--img", "288", "--batch", "24",
         "--workers", "6", "--init", r"ml\weights\model.pt", "--out", r"ml\runs\guncel"])

    print("\n[2/3] Karşılaştırma (aynı doğrulama görselleri):")
    run(["-m", "ml.compare", r"ml\weights", r"ml\runs\guncel"])
    print("\nKamera koşulları testi (yeni model):")
    run(["-m", "ml.stress_test", r"ml\runs\guncel"])

    if input("\n[3/3] Yeni model siteye yüklensin mi? (E/H): ").strip().lower() != "e":
        print(r"Yüklenmedi. Yeni model ml\runs\guncel klasöründe duruyor.")
        return
    backup = ML / "weights_onceki"
    backup.mkdir(exist_ok=True)
    for f in FILES:
        shutil.copy2(ML / "weights" / f, backup / f)
        shutil.copy2(ML / "runs" / "guncel" / f, ML / "weights" / f)
    print(r"Yüklendi (önceki model: ml\weights_onceki). Backend penceresini kapatıp  py baslat.py  ile yeniden açın.")


if __name__ == "__main__":
    main()
