"""ElektroVision'u başlatır: backend (FastAPI + CNN) ve frontend (React) iki ayrı pencerede açılır,
ardından tarayıcıda http://localhost:3000 açılır.

Çalıştırma:  py baslat.py
"""
import shutil
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV_PY = ROOT / "backend" / ".venv" / "Scripts" / "python.exe"


def npm() -> str:
    found = shutil.which("npm")
    return found or r"C:\Program Files\nodejs\npm.cmd"


def main() -> None:
    if not VENV_PY.exists():
        print("Kurulum yapılmamış. Önce:  py kurulum.py")
        sys.exit(1)
    new_window = subprocess.CREATE_NEW_CONSOLE
    subprocess.Popen([str(VENV_PY), "-m", "uvicorn", "server:app", "--host", "127.0.0.1", "--port", "8001"],
                     cwd=ROOT / "backend", creationflags=new_window)
    subprocess.Popen([npm(), "run", "dev"], cwd=ROOT / "frontend", creationflags=new_window)
    print("Sunucular başlatılıyor, tarayıcı birazdan açılacak...")
    print("Kapatmak için açılan iki pencereyi kapatın.")
    time.sleep(10)
    webbrowser.open("http://localhost:3000")


if __name__ == "__main__":
    main()
