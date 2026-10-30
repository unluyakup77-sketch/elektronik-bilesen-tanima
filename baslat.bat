@echo off
rem ElektroVision'u başlatır: backend (FastAPI + CNN) ve frontend (React) iki ayrı pencerede açılır.
chcp 65001 >nul
cd /d "%~dp0"

if not exist "backend\.venv\Scripts\python.exe" (
  echo Kurulum yapilmamis. Once kurulum.bat dosyasini calistirin.
  pause
  exit /b 1
)

start "ElektroVision Backend" /D "%~dp0backend" cmd /k .venv\Scripts\python.exe -m uvicorn server:app --host 127.0.0.1 --port 8001
start "ElektroVision Frontend" /D "%~dp0frontend" cmd /k npm run dev

echo Sunucular baslatiliyor, tarayici birazdan acilacak...
timeout /t 10 >nul
start "" http://localhost:3000
