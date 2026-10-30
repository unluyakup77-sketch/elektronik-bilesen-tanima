@echo off
rem Sitede "Sonuc yanlis mi?" ile kaydedilen fotograflarla modeli ince ayar yapar (NVIDIA GPU, ~10 dk),
rem yeni modeli eskisiyle karsilastirir ve onay verirseniz siteye yukler.
chcp 65001 >nul
cd /d "%~dp0backend"
set PYTHONIOENCODING=utf-8

if not exist ".venv-train\Scripts\python.exe" (
  echo GPU egitim ortami yok. Kurulum icin backend\ml\README.md - "Yeniden egitim" bolumune bakin.
  pause
  exit /b 1
)

echo [1/3] Ince ayar basliyor (mevcut model + kaydettiginiz fotograflar)...
.venv-train\Scripts\python.exe -u -m ml.train --arch convnext_tiny --epochs 8 --workers 6 --init ml\weights\model.pt --out ml\runs\guncel || goto :hata

echo.
echo [2/3] Karsilastirma (ayni dogrulama gorselleri):
.venv-train\Scripts\python.exe -m ml.compare ml\weights ml\runs\guncel || goto :hata
echo.
echo Kamera kosullari testi (yeni model):
.venv-train\Scripts\python.exe -m ml.stress_test ml\runs\guncel || goto :hata

echo.
set /p OK="[3/3] Yeni model siteye yuklensin mi? (E/H): "
if /i not "%OK%"=="E" (
  echo Yuklenmedi. Yeni model ml\runs\guncel klasorunde duruyor.
  pause
  exit /b 0
)
if not exist ml\weights_onceki mkdir ml\weights_onceki
copy /y ml\weights\model.pt ml\weights_onceki\ >nul
copy /y ml\weights\metrics.json ml\weights_onceki\ >nul
copy /y ml\weights\split.json ml\weights_onceki\ >nul
copy /y ml\runs\guncel\model.pt ml\weights\ >nul
copy /y ml\runs\guncel\metrics.json ml\weights\ >nul
copy /y ml\runs\guncel\split.json ml\weights\ >nul
echo Yuklendi (onceki model: ml\weights_onceki). Backend penceresini kapatip baslat.bat ile yeniden acin.
pause
exit /b 0

:hata
echo Hata olustu, yukaridaki mesaji kontrol edin.
pause
exit /b 1
