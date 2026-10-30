@echo off
rem İlk kurulum: Python sanal ortamı + backend paketleri + frontend paketleri.
rem Gereken: Python 3.12 ve Node.js (LTS) kurulu olmalı.
chcp 65001 >nul
cd /d "%~dp0"

set "PY=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if not exist "%PY%" set "PY=python"

echo [1/3] Python sanal ortami olusturuluyor...
if not exist "backend\.venv\Scripts\python.exe" "%PY%" -m venv backend\.venv || goto :hata

echo [2/3] Backend paketleri kuruluyor (PyTorch buyuk, birkac dakika surebilir)...
backend\.venv\Scripts\python.exe -m pip install --upgrade pip || goto :hata
backend\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt || goto :hata

echo [3/3] Frontend paketleri kuruluyor...
cd frontend
call npm install || goto :hata
cd ..

echo.
echo Kurulum tamamlandi. Uygulamayi acmak icin baslat.bat dosyasini calistirin.
pause
exit /b 0

:hata
echo.
echo Kurulum sirasinda hata olustu. Yukaridaki mesaji kontrol edin.
pause
exit /b 1
