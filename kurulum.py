"""İlk kurulum: Python sanal ortamı + backend paketleri + frontend paketleri.
Gereken: Python 3.12 ve Node.js (LTS) kurulu olmalı.

Çalıştırma:  py kurulum.py
"""
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / "backend" / ".venv"
VENV_PY = VENV / "Scripts" / "python.exe"


def run(cmd: list[str], cwd: Path = ROOT) -> None:
    if subprocess.call(cmd, cwd=cwd) != 0:
        print("\nKurulum sırasında hata oluştu. Yukarıdaki mesajı kontrol edin.")
        sys.exit(1)


def main() -> None:
    print("[1/3] Python sanal ortamı oluşturuluyor...")
    if not VENV_PY.exists():
        run([sys.executable, "-m", "venv", str(VENV)])

    print("[2/3] Backend paketleri kuruluyor (PyTorch büyük, birkaç dakika sürebilir)...")
    run([str(VENV_PY), "-m", "pip", "install", "--upgrade", "pip"])
    run([str(VENV_PY), "-m", "pip", "install", "-r", str(ROOT / "backend" / "requirements.txt")])

    print("[3/3] Frontend paketleri kuruluyor...")
    npm = shutil.which("npm") or r"C:\Program Files\nodejs\npm.cmd"
    run([npm, "install"], cwd=ROOT / "frontend")

    print("\nKurulum tamamlandı. Uygulamayı açmak için:  py baslat.py")


if __name__ == "__main__":
    main()
