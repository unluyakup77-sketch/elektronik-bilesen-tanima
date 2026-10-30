"""
CNN eğitim scripti — ImageNet ön-eğitimli bir CNN gövdesi + Transfer Learning (PyTorch).

Çalıştırma:  cd backend && python -m ml.train                          (varsayılan: ConvNeXt-Tiny)
             cd backend && python -m ml.train --arch efficientnet_b0 --epochs 20
Seçenekler:  --arch {convnext_tiny, efficientnet_b0, efficientnet_v2_s}  --epochs N  --img PX  --batch N
             --out KLASOR (varsayılan ml/runs/<arch>; sitenin kullandığı model için: --out ml/weights)
Çıktılar (--out klasörüne):
  model.pt     -> eğitilmiş ağırlıklar (en iyi doğrulama doğruluğu)
  metrics.json -> mimari, epoch geçmişi, karışıklık matrisi, sınıf bazlı metrikler
  split.json   -> doğrulama seti (tekrarlanabilirlik için)

Doğrulama seti: ml/weights/split.json varsa AYNEN kullanılır (eski ve yeni modeller aynı görsellerle
karşılaştırılsın diye); yeni eklenen görseller yalnızca eğitime girer. split.json'da hiç görseli
olmayan sınıfların (ör. sonradan eklenen "bilinmiyor") %20'si doğrulamaya ayrılır.
GPU (CUDA) varsa otomatik kullanılır; yoksa CPU'da (çok daha yavaş) çalışır.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from ml.classes import CLASSES, UNKNOWN

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "dataset"
BASE_SPLIT = ROOT / "weights" / "split.json"
USER_DIR = ROOT / "dataset_user"
VAL_RATIO = 0.2
SEED = 42
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

ARCHS = {
    "efficientnet_b0": "EfficientNet-B0",
    "efficientnet_v2_s": "EfficientNetV2-S",
    "convnext_tiny": "ConvNeXt-Tiny",
}


class RandomLowRes:
    """Görüntüyü rastgele küçültüp geri büyütür: düşük çözünürlüklü / bulanık kamera karelerine dayanıklılık."""

    def __init__(self, p: float = 0.3, min_side: int = 56, max_side: int = 128):
        self.p, self.min_side, self.max_side = p, min_side, max_side

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        s = random.randint(self.min_side, self.max_side)
        w, h = img.size
        k = s / min(w, h)
        if k >= 1:
            return img
        small = img.resize((max(1, int(w * k)), max(1, int(h * k))), Image.BILINEAR)
        return small.resize((w, h), Image.BILINEAR)


def train_transform(img: int) -> transforms.Compose:
    return transforms.Compose([
        RandomLowRes(),
        transforms.RandomResizedCrop(img, scale=(0.5, 1.0), ratio=(0.75, 1.33)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(25),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
    ])


def eval_transform(img: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(img),
        transforms.CenterCrop(img),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


# geriye dönük uyumluluk (eski kod eval_tf kullanıyor)
eval_tf = eval_transform(224)


class ImageList(Dataset):
    def __init__(self, items: list[tuple[str, int]], tf: transforms.Compose):
        self.items, self.tf = items, tf

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        path, label = self.items[i]
        return self.tf(Image.open(path).convert("RGB")), label


class CameraSimList(ImageList):
    """Eğitim örneklerinin bir kısmını "kamera görüntüsüne" çevirir (bkz. ml/camera_sim.py):
    bileşen arka planı silinip masa/el/kumaş fotoğrafının üzerine küçük ve rastgele konumda yapıştırılır,
    ardından bulanıklık/JPEG/ışık bozulması eklenir. "bilinmiyor" görselleri de bozulmaya uğrar."""

    def __init__(self, items, tf, backgrounds: list[str], unknown_idx: int | None, p: float = 0.6):
        super().__init__(items, tf)
        self.backgrounds, self.unknown_idx, self.p = backgrounds, unknown_idx, p

    def __getitem__(self, i: int):
        from ml.camera_sim import composite, degrade

        path, label = self.items[i]
        img = Image.open(path).convert("RGB")
        rng = random.Random()
        if self.backgrounds and rng.random() < self.p:
            if label != self.unknown_idx:
                img = composite(img, Image.open(rng.choice(self.backgrounds)), rng)
            img = degrade(img, rng)
        return self.tf(img), label


def build_model(num_classes: int, pretrained: bool = True, arch: str = "efficientnet_b0") -> nn.Module:
    if arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        m.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.classifier[1].in_features, num_classes))
    elif arch == "efficientnet_v2_s":
        m = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None)
        m.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.classifier[1].in_features, num_classes))
    elif arch == "convnext_tiny":
        m = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None)
        m.classifier[2] = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.classifier[2].in_features, num_classes))
    else:
        raise ValueError(f"bilinmeyen mimari: {arch}")
    return m


def head_parameters(model: nn.Module) -> list[nn.Parameter]:
    return list(model.classifier.parameters())


def split_dataset() -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    """Sabit doğrulama seti: BASE_SPLIT'teki dosyalar doğrulamada kalır, geri kalan her şey eğitime gider."""
    fixed: set[tuple[str, str]] = set()
    if BASE_SPLIT.exists():
        fixed = {(Path(p).parent.name, Path(p).name) for p in json.loads(BASE_SPLIT.read_text(encoding="utf-8"))["val"]}
    fixed_classes = {c for c, _ in fixed}
    rng = random.Random(SEED)
    train, val = [], []
    for ci, cls in enumerate(CLASSES):
        files = sorted(str(p) for p in (DATA_DIR / cls).glob("*.jpg"))
        if cls in fixed_classes:
            for f in files:
                (val if (cls, Path(f).name) in fixed else train).append((f, ci))
        else:
            rng.shuffle(files)
            n_val = max(1, int(len(files) * VAL_RATIO))
            val += [(f, ci) for f in files[:n_val]]
            train += [(f, ci) for f in files[n_val:]]
    return train, val


def user_items(repeat: int) -> list[tuple[str, int]]:
    """Sitedeki "Sonuç yanlış mı?" ile toplanan gerçek kamera/telefon fotoğrafları (ml/dataset_user/<sinif>/).
    Gerçek kullanım koşullarını en iyi bunlar temsil ettiği için eğitime `repeat` kez girerler (doğrulamaya girmezler)."""
    out = []
    for ci, cls in enumerate(CLASSES):
        out += [(str(p), ci) for p in sorted((USER_DIR / cls).glob("*.jpg"))] * repeat
    return out


def amp_dtype(dev: str) -> torch.dtype:
    # bfloat16, float16 gibi taşmaz (ConvNeXt'te fp16 bazı batch'lerde NaN üretiyordu); destek yoksa float16
    return torch.bfloat16 if dev == "cuda" and torch.cuda.is_bf16_supported() else torch.float16


def evaluate(model: nn.Module, loader: DataLoader, crit: nn.Module, dev: str):
    model.eval()
    n, loss_sum, preds, labels = 0, 0.0, [], []
    with torch.no_grad(), torch.autocast(dev, dtype=amp_dtype(dev), enabled=dev == "cuda"):
        for x, y in loader:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            # test-time augmentation: orijinal + ayna görüntüsü (sunucudaki çıkarım ile aynı)
            out = (model(x).float().softmax(1) + model(torch.flip(x, dims=[3])).float().softmax(1)) / 2
            loss_sum += crit(out.clamp_min(1e-8).log(), y).item() * len(y)
            n += len(y)
            preds += out.argmax(1).tolist()
            labels += y.tolist()
    preds_a, labels_a = np.array(preds), np.array(labels)
    return loss_sum / n, float((preds_a == labels_a).mean()), preds_a, labels_a


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="convnext_tiny", choices=list(ARCHS))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--img", type=int, default=224)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=None)
    ap.add_argument("--camera-sim", type=float, default=0.6, help="eğitim örneğinin kamera benzetimine girme olasılığı (0 = kapalı)")
    ap.add_argument("--init", default=None, help="başlangıç ağırlıkları (ör. ml/weights/model.pt) — ince ayar için")
    ap.add_argument("--user-repeat", type=int, default=3, help="ml/dataset_user fotoğraflarının eğitimde kaç kez tekrarlanacağı")
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else ROOT / "runs" / args.arch
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(SEED)
    random.seed(SEED)

    train_items, val_items = split_dataset()
    user = user_items(args.user_repeat)
    if user:
        print(f"kullanıcı fotoğrafı: {len(user) // args.user_repeat} (x{args.user_repeat})", flush=True)
        train_items += user
    counts = Counter(l for _, l in train_items + val_items)
    print(f"cihaz={dev} mimari={args.arch} eğitim={len(train_items)} doğrulama={len(val_items)}", flush=True)

    # sınıf dengesizliği için ağırlıklı kayıp
    tc = Counter(l for _, l in train_items)
    w = torch.tensor([len(train_items) / (len(CLASSES) * max(1, tc[i])) for i in range(len(CLASSES))], dtype=torch.float)

    loader_kw = dict(num_workers=args.workers, pin_memory=dev == "cuda", persistent_workers=args.workers > 0)
    unknown_idx = CLASSES.index(UNKNOWN) if UNKNOWN in CLASSES else None
    # arka plan havuzu: ml/backgrounds + eğitimdeki "bilinmiyor" görselleri (doğrulamadakiler kullanılmaz)
    backgrounds = [str(p) for p in sorted((ROOT / "backgrounds").glob("*.jpg"))] + [p for p, l in train_items if l == unknown_idx]
    if args.camera_sim > 0:
        print(f"kamera benzetimi: olasılık={args.camera_sim} arka plan={len(backgrounds)}", flush=True)
        train_ds = CameraSimList(train_items, train_transform(args.img), backgrounds, unknown_idx, p=args.camera_sim)
    else:
        train_ds = ImageList(train_items, train_transform(args.img))
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=True, **loader_kw)
    val_dl = DataLoader(ImageList(val_items, eval_transform(args.img)), batch_size=64, **loader_kw)

    model = build_model(len(CLASSES), arch=args.arch)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location="cpu"))
        print("başlangıç ağırlıkları:", args.init, flush=True)
    model = model.to(dev)
    crit = nn.CrossEntropyLoss(weight=w.to(dev), label_smoothing=0.1)
    val_crit = nn.NLLLoss()
    head = head_parameters(model)
    head_ids = {id(p) for p in head}
    backbone = [p for p in model.parameters() if id(p) not in head_ids]
    # tüm ağ ince ayar yapılır; gövde ön-eğitimli olduğu için daha küçük öğrenme oranı alır
    opt = torch.optim.AdamW([{"params": backbone, "lr": 2e-4}, {"params": head, "lr": 2e-3}], weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[2e-4, 2e-3], total_steps=args.epochs * len(train_dl), pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=dev == "cuda" and amp_dtype(dev) == torch.float16)

    history, best_acc, best = [], 0.0, None
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, n, loss_sum, correct = time.time(), 0, 0.0, 0
        # ilk epoch: sadece sınıflandırıcı başlık eğitilir (rastgele başlık gövdeyi bozmasın)
        for p in backbone:
            p.requires_grad = ep > 1
        for x, y in train_dl:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(dev, dtype=amp_dtype(dev), enabled=dev == "cuda"):
                out = model(x)
                loss = crit(out.float(), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            if not torch.isfinite(loss):  # bu adım atlandı; ortalamayı bozmasın
                continue
            loss_sum += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            n += len(y)
        vl, va, preds, labels = evaluate(model, val_dl, val_crit, dev)
        row = {"epoch": ep, "train_loss": round(loss_sum / n, 4), "train_acc": round(correct / n, 4), "val_loss": round(vl, 4), "val_acc": round(va, 4)}
        history.append(row)
        print(row, f"{time.time() - t0:.0f}s", flush=True)
        if va >= best_acc:
            best_acc, best = va, (preds, labels)
            torch.save(model.state_dict(), out_dir / "model_new.pt")

    preds, labels = best
    k = len(CLASSES)
    cm = np.zeros((k, k), dtype=int)
    for t, p in zip(labels, preds):
        cm[t, p] += 1
    per_class = []
    for i, c in enumerate(CLASSES):
        tp = cm[i, i]
        prec = tp / cm[:, i].sum() if cm[:, i].sum() else 0.0
        rec = tp / cm[i, :].sum() if cm[i, :].sum() else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_class.append({"key": c, "precision": round(float(prec), 4), "recall": round(float(rec), 4), "f1": round(float(f1), 4), "support": int(cm[i, :].sum()), "total_images": int(counts[i])})

    metrics = {
        "arch": args.arch,
        "architecture": f"{ARCHS[args.arch]} (ImageNet ön-eğitimli) + Transfer Learning (tüm ağ ince ayar, 1. epoch sadece sınıflandırıcı)",
        "framework": f"PyTorch {torch.__version__}",
        "device": torch.cuda.get_device_name(0) if dev == "cuda" else "CPU",
        "input_size": args.img,
        "epochs": args.epochs,
        "batch_size": args.batch,
        "optimizer": "AdamW (gövde 2e-4, başlık 2e-3) + OneCycleLR, karma hassasiyet (AMP)",
        "loss": "Ağırlıklı CrossEntropy (label smoothing 0.1)",
        "augmentation": "RandomResizedCrop, çevirme, ±25° döndürme, TrivialAugmentWide, rastgele düşük çözünürlük, RandomErasing"
        + (f" + kamera benzetimi (%{args.camera_sim * 100:.0f}: arka planı silinip masa/el fotoğrafına yapıştırma, bulanıklık, JPEG, ışık)" if args.camera_sim > 0 else ""),
        "init": args.init,
        "train_size": len(train_items),
        "val_size": len(val_items),
        "best_val_acc": round(best_acc, 4),
        "classes": CLASSES,
        "history": history,
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    # ağırlıklar ve metrikler birlikte güncellenir (sunucu her zaman eşleşen bir çift görür)
    (out_dir / "model_new.pt").replace(out_dir / "model.pt")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "split.json").write_text(json.dumps({"val": [str(Path(p).relative_to(ROOT)) for p, _ in val_items]}, ensure_ascii=False), encoding="utf-8")
    print("En iyi doğrulama doğruluğu:", best_acc)


if __name__ == "__main__":
    main()
