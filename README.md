# ElektroVision — CNN ile Elektronik Bileşen Tanıma

Fotoğraf veya canlı kamera görüntüsünden elektronik bileşeni tanıyan ve ne işe yaradığını anlatan web uygulaması.
Kararı Python + PyTorch ile eğitilmiş bir CNN (ConvNeXt-Tiny, transfer learning, 19 bileşen sınıfı + "bileşen değil") verir.
Doğrulama setinde bileşen doğruluğu %95,3 (önceki model: %91,6).
Model ve eğitim ayrıntıları: [backend/ml/README.md](backend/ml/README.md).

## Kurulum ve çalıştırma (Windows)

Gerekenler: **Python 3.12** ve **Node.js (LTS)**.

1. `kurulum.bat` — ilk seferde bir kez çalıştırılır (Python sanal ortamı, PyTorch vb. paketler, frontend paketleri).
2. `baslat.bat` — backend ve frontend'i iki pencerede açar, tarayıcıda `http://localhost:3000` açılır.

İlk açılışta backend'in hazır olması PyTorch yüklendiği için 30–60 sn sürebilir.
Kapatmak için açılan iki komut penceresini kapatın.

Elle çalıştırmak isterseniz:

```bash
cd backend && .venv\Scripts\python.exe -m uvicorn server:app --port 8001   # API: http://localhost:8001/api
cd frontend && npm run dev                                                  # Site: http://localhost:3000
```

## Yapı

```
backend/
  server.py            FastAPI uygulaması (tüm uçlar /api altında)
  routers/scan.py      tahmin, geçmiş, direnç, model bilgisi uçları
  ml/                  CNN: veri seti, eğitim (train.py), çıkarım (inference.py), ağırlıklar (weights/)
  lib/components_info.py  19 bileşenin Türkçe bilgi bankası
  lib/resistor_cv.py   direnç renk bantlarını OpenCV ile okuma
  lib/db.py            tarama geçmişi (SQLite: backend/data/elektrovision.db)
  lib/verifier.py      isteğe bağlı "Çift kontrol" (Gemini) — varsayılan kapalı
frontend/              React + TypeScript arayüz (Tara, Bileşen Kütüphanesi, Model & Metrikler sayfaları)
```

## Ayarlar (isteğe bağlı)

`backend/.env.example` dosyasını `backend/.env` olarak kopyalayın. Hiçbir ayar zorunlu değildir;
`GEMINI_API_KEY` girilirse "Gelişmiş ayarlar → Çift kontrol" Gemini ile ikinci görüş alır, girilmezse karar yalnızca CNN'indir.

## Modeli yeniden eğitme

NVIDIA ekran kartı ile ~20 dk sürer (CPU'da saatler). Adımlar ve GPU ortamının kurulumu:
[backend/ml/README.md → Yeniden eğitim](backend/ml/README.md#yeniden-eğitim-gpu).
