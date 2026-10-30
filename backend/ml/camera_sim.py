"""
Kamera benzetimi (sentetik "gerçek dünya" görüntüleri).

Veri setindeki görsellerin çoğu beyaz arka planlı, bileşenin kadrajı doldurduğu ürün fotoğraflarıdır.
Gerçek kamerada ise parça küçük, arkada masa/el/kumaş var, görüntü bulanık ve ışık kötü. Bu modül
eğitim sırasında bu koşulları üretir:
  1. remove_white_background: kenara bağlı beyaz/açık gri pikselleri saydam yapar (ürün fotoğrafı -> kesik parça)
  2. composite: parçayı rastgele boyut/konum/açıyla bir arka plan fotoğrafının üzerine yapıştırır (+ hafif gölge)
  3. degrade: bulanıklık, JPEG sıkıştırma, parlaklık/renk sıcaklığı, gürültü
"""
from __future__ import annotations

import io
import random

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def remove_white_background(img: Image.Image) -> Image.Image:
    """Kenarlara bağlı açık renkli arka planı saydam yapar. Arka plan beyaz değilse görüntü olduğu gibi (opak) döner."""
    import cv2

    rgb = np.asarray(img.convert("RGB"))
    h, w = rgb.shape[:2]
    light = (rgb.min(axis=2) > 205) & (rgb.max(axis=2).astype(int) - rgb.min(axis=2) < 30)
    n, lab = cv2.connectedComponents(light.astype(np.uint8), connectivity=4)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    bg = np.isin(lab, list(border)) & light
    frac = bg.mean()
    alpha = np.full((h, w), 255, np.uint8)
    # gerçekten beyaz arka planlı bir ürün fotoğrafı; parçanın kendisi de beyaz/şeffafsa (cam sigorta, seramik)
    # silme sonrası çok az piksel kalır — o zaman parça kaybolmasın diye görüntü olduğu gibi kullanılır
    if 0.15 < frac < 0.88:
        alpha[bg] = 0
        alpha = cv2.GaussianBlur(alpha, (3, 3), 0)
    out = img.convert("RGBA")
    out.putalpha(Image.fromarray(alpha))
    return out


def composite(fg: Image.Image, bg: Image.Image, rng: random.Random, scale: tuple[float, float] = (0.25, 0.85), size: int = 320) -> Image.Image:
    """Parçayı (fg) arka plan (bg) üzerine yapıştırır. scale: parçanın uzun kenarının kadraja oranı."""
    canvas = bg.convert("RGB")
    # arka plandan rastgele kare bir bölge al (kamera kadrajı gibi)
    side = min(canvas.size)
    cs = int(side * rng.uniform(0.6, 1.0))
    x0, y0 = rng.randint(0, canvas.width - cs), rng.randint(0, canvas.height - cs)
    canvas = canvas.crop((x0, y0, x0 + cs, y0 + cs)).resize((size, size), Image.BILINEAR)

    part = remove_white_background(fg) if fg.mode != "RGBA" else fg
    part = part.rotate(rng.uniform(-180, 180), expand=True, resample=Image.BILINEAR)
    s = int(size * rng.uniform(*scale))
    part.thumbnail((s, s), Image.BILINEAR)
    x = rng.randint(0, max(0, size - part.width))
    y = rng.randint(0, max(0, size - part.height))
    # yumuşak gölge: parçanın masaya oturduğu hissini verir
    shadow_alpha = part.getchannel("A").filter(ImageFilter.GaussianBlur(4)).point(lambda a: int(a * 0.35))
    shadow = Image.new("RGBA", part.size, (0, 0, 0, 0))
    shadow.putalpha(shadow_alpha)
    canvas.paste(shadow, (x + 3, y + 4), shadow)
    canvas.paste(part, (x, y), part)
    return canvas


def degrade(img: Image.Image, rng: random.Random, strength: float = 1.0) -> Image.Image:
    """Ucuz web kamerası / telefon görüntüsü bozulmaları."""
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(1 - 0.4 * strength, 1 + 0.3 * strength))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(1 - 0.3 * strength, 1 + 0.2 * strength))
    img = ImageEnhance.Color(img).enhance(rng.uniform(1 - 0.4 * strength, 1 + 0.2 * strength))
    if rng.random() < 0.5:  # renk sıcaklığı (sarı ampul / mavi gün ışığı)
        r, g, b = img.split()
        k = rng.uniform(-0.12, 0.12) * strength
        r = r.point(lambda v: min(255, int(v * (1 + k))))
        b = b.point(lambda v: min(255, int(v * (1 - k))))
        img = Image.merge("RGB", (r, g, b))
    if rng.random() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.6) * strength))
    if rng.random() < 0.4:
        a = np.asarray(img).astype(np.int16)
        a = a + np.random.default_rng(rng.randint(0, 1 << 30)).normal(0, 6 * strength, a.shape).astype(np.int16)
        img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=rng.randint(35, 85))
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
