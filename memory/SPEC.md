# ElektroVision — SPEC

Turkish-language web app for an EEE "Derin Öğrenme" course project: upload/take a photo of an electronic component → a PyTorch CNN (ConvNeXt-Tiny transfer learning, 19 component classes + `bilinmiyor` "not a component" class; architecture read from metrics.json `arch`) identifies it. The decision is CNN-only; Gemini 3 Flash verifies ONLY when the user enables "Çift kontrol" (hidden under 'Gelişmiş ayarlar', default off). If label is direnc, bands are read with classical OpenCV (lib/resistor_cv.py, conservative: returns detected=false + value=null when unsure) and Python (lib/resistor.py) computes ohms; user can edit bands in UI (GET /resistor/value). MCU classes (arduino, esp32, stm32) have how_it_works + specs. Live camera dialog (getUserMedia): continuous CNN-only recognition every ~1.2s via POST /predict/live (center square crop, not stored) + 'Kareyi Tara' for full scan. Result shows name, confidence, source badge, CNN top-3, AI note, "Ne işe yarar?", usage areas, identification tips, circuit symbol (inline SVG).

## Auth
None — public app, no login.

## Local run (Windows)
py kurulum.py (venv + pip + npm install) then py baslat.py. Storage is SQLite (backend/data/elektrovision.db, lib/db.py) — no MongoDB. Çift kontrol calls the Gemini REST API directly with GEMINI_API_KEY from backend/.env (see .env.example); without a key it falls back to the CNN with a note.

## Classes (keys)
direnc, led, seramik_kondansator, elektrolitik_kondansator, transistor, diyot, entegre, potansiyometre, bobin, role, sigorta, transformator, buzzer, kristal_osilator, buton, display_7seg, arduino, esp32, stm32. Model's class order comes from ml/weights/metrics.json. AI may also return `bilinmiyor` (not a component) → `is_component=false`, name "Tanımlanamadı".

## API (all under /api)
- GET /components → ComponentInfo[] (16, static Turkish KB in backend/lib/components_info.py)
- POST /predict (multipart: file, double_check bool form field) → Prediction (stored in SQLite `predictions`, includes base64 thumbnail). 400 empty, 413 >10MB, 415 not an image.
- GET /predictions → last 24 Prediction, newest first; DELETE /predictions/{id} (404 if missing)
- GET /resistor/value?bands=c1,c2,...  → ResistorValue (400 invalid)
- GET /model → ModelInfo from backend/ml/weights/metrics.json (history, confusion matrix, per-class metrics)
- GET /samples → Sample[] ; GET /samples/{file}.jpg → sample image (one per class, from validation split)

## ML
backend/ml/: classes.py, build_dataset.py (Kaggle + HF + DuckDuckGo images, CLIP-cleaned), expand_dataset.py (adds w2_ images, builds bilinmiyor), train.py (GPU, --arch), compare.py, analyze.py, inference.py, make_samples.py, weights/ (live model), weights_v1/ (old B0). GPU training uses backend/.venv-train (torch cu126). See backend/ml/README.md.

## Frontend routes
/ (scan + history), /kutuphane (library with search + category filters + detail dialog), /model (metrics dashboard).
