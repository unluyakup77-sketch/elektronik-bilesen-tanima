# ElektroVision — CNN ile Elektronik Bileşen Tanıma

Bu klasör projenin **derin öğrenme** kısmını içerir (Python + PyTorch).

## Dosyalar
| Dosya | Görev |
|---|---|
| `classes.py` | Sınıf listesi (19 bileşen + `bilinmiyor`), arama sorguları, CLIP temizlik metinleri |
| `build_dataset.py` | İlk veri seti: Kaggle + HuggingFace + web görselleri → CLIP ile temizlik → `dataset/<sinif>/` |
| `expand_dataset.py` | Veri setini **silmeden genişletir**: zayıf sınıflara web görseli ekler, CLIP ile süzer, tekrarları atar; `bilinmiyor` sınıfını oluşturur |
| `train.py` | Transfer learning ile eğitim (ConvNeXt-Tiny / EfficientNet-B0 / EfficientNetV2-S), GPU + karma hassasiyet |
| `compare.py` | Birden çok modeli **aynı doğrulama görsellerinde** karşılaştırır (doğruluk, yanlış ret, CPU hızı) |
| `analyze.py` | Hata analizi: en sık karışan sınıf çiftleri + her çift için görsel kolaj (`analysis/`) |
| `inference.py` | Sunucunun kullandığı çıkarım: modeli `weights/metrics.json`'daki mimariyle yükler |
| `make_samples.py` | Doğrulama setinden her sınıf için örnek görsel seçer (sitedeki "Örnek görsellerle dene") |
| `weights/` | Sitenin kullandığı model (`model.pt`, `metrics.json`, `split.json`) |
| `weights_v1/` | Önceki model (EfficientNet-B0, %91,6) — karşılaştırma için saklanıyor |

## Sınıflar (19 + 1)
direnç, LED, seramik kondansatör, elektrolitik kondansatör, transistör, diyot, entegre (IC),
potansiyometre, bobin, röle, sigorta, transformatör, buzzer, kristal osilatör, buton, 7-segment display,
Arduino, ESP32, STM32 **+ `bilinmiyor` (bileşen değil)**: masa, el, klavye, kablo gibi görüntüler.
Canlı kamerada kadrajda bileşen yokken modelin rastgele bir bileşen adı söylemesini engeller.

## Model (v2)
- **Mimari:** ConvNeXt-Tiny gövde (4 aşama, 7×7 derinlemesine evrişim blokları, LayerNorm, GELU) + GlobalAvgPool + Dropout(0.3) + Dense(768→20)
- **Transfer learning:** ImageNet ön-eğitimli; 1. epoch'ta sadece sınıflandırıcı, sonra tüm ağ ince ayar (gövde lr 2e-4, başlık 2e-3)
- **Veri artırma:** RandomResizedCrop, yatay/dikey çevirme, ±25° döndürme, TrivialAugmentWide, rastgele düşük çözünürlük, RandomErasing
- **Kayıp:** sınıf ağırlıklı CrossEntropy + label smoothing 0.1 · **Optimizer:** AdamW + OneCycleLR · bfloat16 karma hassasiyet
- **Çıkarım:** test-time augmentation (orijinal + ayna görüntüsü ortalaması)
- **Eğitim:** 25 epoch, NVIDIA RTX 3050 Ti üzerinde ~20 dk

### v1 → v2 karşılaştırması (aynı 1063 bileşen doğrulama görseli)
| Model | Bileşen doğruluğu | Bileşene "bilinmiyor" deme | Boş kadrajı tanıma | CPU süresi |
|---|---|---|---|---|
| v1 EfficientNet-B0 (eski veri, 9 epoch) | %91,6 | – | yok | ~40 ms |
| v2 EfficientNet-B0 (yeni veri + tarif) | %94,8 | %0,4 | %96,7 | ~45 ms |
| **v2 ConvNeXt-Tiny (kullanılan)** | **%95,3** | **%0,1** | **%97,8** | ~85–105 ms |

En büyük kazanç: röle → transformatör karışıklığı 11 hatadan 0'a indi.
Not: en iyi epoch doğrulama doğruluğuna göre seçildiği için sonuçlar hafif iyimserdir; gerçek kamera
fotoğraflarında başarı daha düşük olabilir.

### Veri seti notları
- Kaggle görselleri 128 px, web görselleri 256 px; eğitimdeki **rastgele düşük çözünürlük** artırması modelin
  çözünürlüğe göre değil parçanın kendisine göre karar vermesine yardım eder.
- `expand_dataset.py` ile eklenen görseller `w2_` önekiyle başlar. Transformatör eklemeleri elle de
  ayıklandı (şema/infografik/şebeke trafosu görselleri silindi).
- Doğrulama seti `weights/split.json` ile **sabittir**; yeni görseller yalnızca eğitime girer.

## v3: Gerçek kamera koşulları (kamera benzetimi)
v2 eğitim tarzı fotoğraflarda %96 doğruyken, parçanın masanın üzerinde küçük göründüğü kamera
görüntülerinde **%9–28'e düşüyordu** (çoğunlukla "bileşen değil" diyordu). Sebep: veri setindeki görsellerin
çoğu beyaz arka planlı, parçanın kadrajı doldurduğu ürün fotoğrafları.

Çözüm `camera_sim.py`: eğitim örneklerinin %60'ında parçanın beyaz arka planı silinir, rastgele boyut/açı ile
masa, el, kesim matı, breadboard gibi 745 arka plan fotoğrafının (`backgrounds/`) üzerine yapıştırılır,
ardından bulanıklık, JPEG, gürültü ve ışık bozulması eklenir. Ayrıca "sigorta" sınıfına otomatik sigorta (MCB)
görselleri eklendi (Türkçede ikisine de sigorta denir; önceki veri setinde yalnızca cam/bıçak sigorta vardı).

`stress_test.py` sonuçları (doğrulama görselleri, eğitimde kullanılmayan arka planlar):

| Koşul | v2 | **v3** |
|---|---|---|
| Orijinal fotoğraf | %96,0 | **%96,3** |
| Masada, kadrajın %70'i | %28,0 | **%83,5** |
| Masada, kadrajın %50'si | %23,5 | **%82,8** |
| Masada, kadrajın %35'i (uzak) | %9,0 | **%69,3** |
| Arka planı silinmiş parça masada, %60 | %13,8 | **%77,7** |

## v5 (kullanılan model): sınıf bazında ≥%80 hedefi
- Zayıf 8 sınıfa gerçek ortam fotoğrafları eklendi (elde, breadboard'da, masada); reddedilen görseller `dataset_blacklist.txt`.
- ImageNet'ten 30 epoch kamera benzetimiyle (%70) eğitim, ardından **288×288** çözünürlükte ince ayar.
- Çıkarımda 6 görünümlü TTA: merkez + yakınlaştırılmış merkez + kırpmasız tüm görüntü (her biri + ayna). Canlıda 4 görünüm.
- Ağırlıklar yarı hassasiyetle (fp16) saklanır: 56 MB, doğruluk değişmez.

`python -m ml.stress_test` (kamera koşulları ortalaması, sınıf bazında rapor): v3 %76,7 → **v5 %86,0**; orijinal fotoğraf %95,8.
Hedefin altında kalanlar: transformatör, röle, sigorta, diyot. Röle/transformatör hatalarının çoğu Kaggle'daki
endüstriyel parçalardan (kontaktör, katı hal rölesi, trafo merkezi); Kaggle hariç röle %80,2, diyot %83,6.

## Otomatik AI desteği (açıkça belirtilir)
CNN güveni %50'nin altındaysa ve `backend/.env` içinde `GEMINI_API_KEY` tanımlıysa Gemini ikinci görüş verir;
sonuç "CNN + AI" rozeti ve "CNN emin olmadığı için AI desteği alındı" notuyla gösterilir. Anahtar yoksa karar CNN'de kalır.
Test: `python -m pytest tests/test_ai_fallback.py -n 0`.

## Kullanıcı geri bildirimiyle öğrenme (insan-döngüde)
Sitede her sonucun altında **"Sonuç doğru mu?"** kartı var. Onaylanan/düzeltilen fotoğraflar
`ml/dataset_user/<sinif>/` altına kaydedilir ve sonraki eğitimlerde (×3 ağırlıkla) kullanılır.
Projenin kök klasöründeki **`modeli_guncelle.bat`**: mevcut modelden ince ayar yapar (~8 dk), eski ve yeni modeli
karşılaştırır, onay verilirse yeni modeli yükler (öncekini `ml/weights_onceki/` içine yedekler).
En iyi sonuç için: hocaya göstereceğiniz her parçanın kendi kameranızla 10–20 fotoğrafını tarayıp onaylayın.

## Veri setini oluşturma
Görseller telif hakkı sahiplerine ait olduğu için **depoda yer almaz** (`dataset/`, `backgrounds/`).
Eğitilmiş model (`weights/model.pt`) depodadır; uygulamayı çalıştırmak için veri setine gerek yoktur.
Veri setini yeniden oluşturmak için (internet gerekir, ~1 saat):

```bash
cd backend
.venv-train\Scripts\python.exe -m ml.build_dataset --more        # Kaggle/HF (varsa) + web görselleri, CLIP ile temizlik
.venv-train\Scripts\python.exe -m ml.expand_dataset              # zayıf sınıflar + "bilinmiyor" sınıfı
.venv-train\Scripts\python.exe -m ml.expand_dataset --round3     # gerçek ortam fotoğrafları
.venv-train\Scripts\python.exe -m ml.expand_dataset --backgrounds  # kamera benzetimi arka planları
```

Web aramaları zamanla farklı sonuç döndürdüğü için yeniden oluşturulan veri seti birebir aynı olmaz.
Elle reddedilen görseller `dataset_blacklist.txt` ile dışarıda tutulur.

## Yeniden eğitim (GPU)
Eğitim için ayrı bir ortam kullanılır (sunucunun CPU ortamı bozulmasın):

```bash
cd backend
python -m venv .venv-train
.venv-train\Scripts\python.exe -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu126
.venv-train\Scripts\python.exe -m pip install pillow numpy open_clip_torch ddgs requests
.venv-train\Scripts\python.exe -m ml.expand_dataset                 # (isteğe bağlı) veri setini genişlet
.venv-train\Scripts\python.exe -m ml.train --arch convnext_tiny --epochs 25    # -> ml/runs/convnext_tiny
.venv\Scripts\python.exe -m ml.compare ml/weights ml/runs/convnext_tiny        # eski modelle karşılaştır
```

Yeni model daha iyiyse `ml/runs/<isim>/` içindeki `model.pt`, `metrics.json`, `split.json` dosyalarını
`ml/weights/` içine kopyalayın, `python -m ml.make_samples` çalıştırın ve backend'i yeniden başlatın.
Not: `open_clip_torch` kurulurken pip torch'u CPU sürümüyle değiştirebilir; kurulumdan sonra
`python -c "import torch; print(torch.cuda.is_available())"` ile kontrol edin, `False` ise torch satırını
`--force-reinstall --no-deps` ile tekrar çalıştırın.

## Karar mekanizması (backend/routers/scan.py)
1. CNN 20 sınıf için softmax olasılığı verir; **en yüksek olasılıklı sınıf (argmax) karardır.**
2. Sonuç `bilinmiyor` ise "Tanımlanamadı" gösterilir; canlı kamerada "Aranıyor…" yazısı kalır.
3. CNN güveni %60'ın altındaysa kullanıcıya daha net fotoğraf çekmesi önerilir ve diğer olası tahminler gösterilir.
4. (İsteğe bağlı, varsayılan kapalı) "Gelişmiş ayarlar → Çift kontrol" açılırsa ve `GEMINI_API_KEY` tanımlıysa Gemini ikinci görüş verir.

## Direnç renk kodu — klasik görüntü işleme (backend/lib/resistor_cv.py, OpenCV + NumPy)
1. Arka plan rengi kenar piksellerinden bulunur, beyaz dengesi yapılır.
2. Arka plandan farklı pikseller (Otsu eşikleme) → en büyük kontur = direnç; `minAreaRect` ile açı bulunup görüntü döndürülür.
3. Bacaklar kesilir (maske kalınlığı), gövdenin orta şeridinden sütun bazlı medyan renk profili çıkarılır.
4. Gövde renginden belirgin farklı sütun grupları = bantlar; her bant Lab renk uzayında 12 referans renge uzaklıkla sınıflandırılır.
5. Rol kuralları (rakam bantları altın/gümüş olamaz vb.) ve iki okuma yönü denenir, en düşük toplam uzaklık seçilir.
6. **Yanlış değer göstermemek için** renkler referanslara yeterince yakın değilse veya bant sayısı 4/5 değilse otomatik sonuç verilmez; kullanıcı bantları seçer. Ohm değeri `backend/lib/resistor.py` ile hesaplanır: `değer = rakamlar × 10^çarpan`, tolerans son banttan.

## Canlı kamera (gerçek parça tanıma)
Sitedeki "Canlı Kamera" tarayıcı kamerasını açar. Canlı tanıma açıkken ekrandaki kare çerçevenin içi ~1.2 sn'de bir `POST /api/predict/live` ile CNN'e gönderilir (kayıt tutulmaz); aynı sonuç art arda 2 kez ve %60 üzeri güvenle gelince bileşen adı gösterilir. "Kareyi Tara" detaylı sonucu (renk kodu, açıklama) üretir.
