from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel


class ComponentInfo(BaseModel):
    key: str
    name: str
    english_name: str
    category: str
    summary: str
    function: str
    usage_areas: list[str]
    identify_tips: list[str]
    typical_values: str
    packages: list[str]
    how_it_works: Optional[str] = None  # sadece mikrodenetleyici kartları
    specs: list[str] = []


class ResistorValue(BaseModel):
    ohms: float
    formatted: str
    tolerance_percent: float
    min_ohms: float
    max_ohms: float
    formula: str


class ResistorReading(BaseModel):
    bands: list[str]  # okuma sırasına göre renk anahtarları (black, brown, ... gold, silver)
    detected: bool  # renkler fotoğraftan otomatik okunabildi mi
    note: Optional[str] = None
    value: Optional[ResistorValue] = None


class ClassProb(BaseModel):
    key: str
    name: str
    probability: float


class Prediction(BaseModel):
    id: str
    created_at: datetime
    label: str  # bileşen anahtarı veya "bilinmiyor"
    name: str
    confidence: float
    is_component: bool
    source: Literal["cnn", "cnn+ai"]
    cnn_label: str
    cnn_confidence: float
    cnn_top3: list[ClassProb]
    ai_used: bool
    ai_label: Optional[str] = None
    ai_confidence: Optional[float] = None
    ai_note: Optional[str] = None
    agreement: Optional[bool] = None
    thumbnail: str
    info: Optional[ComponentInfo] = None
    resistor: Optional[ResistorReading] = None
    feedback_label: Optional[str] = None  # kullanıcının bildirdiği doğru etiket


class FeedbackIn(BaseModel):
    label: str


class FeedbackResult(BaseModel):
    total: int
    per_class: dict[str, int]


class EpochRow(BaseModel):
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float


class ClassMetric(BaseModel):
    key: str
    name: str
    precision: float
    recall: float
    f1: float
    support: int
    total_images: int


class ModelInfo(BaseModel):
    available: bool
    arch: str = "efficientnet_b0"
    architecture: str
    device: Optional[str] = None
    augmentation: Optional[str] = None
    framework: str
    input_size: int
    epochs: int
    batch_size: int
    optimizer: str
    loss: str
    train_size: int
    val_size: int
    best_val_acc: float
    classes: list[str]
    class_names: list[str]
    history: list[EpochRow]
    confusion_matrix: list[list[int]]
    per_class: list[ClassMetric]
    trained_at: str
    ai_policy: str
    ai_model: str


class Sample(BaseModel):
    key: str
    name: str
    url: str


class LivePrediction(BaseModel):
    label: str
    name: str
    confidence: float
    top3: list[ClassProb]
    probs: list[ClassProb] = []  # tüm sınıflar — arayüz son birkaç kareyi bununla ortalar
