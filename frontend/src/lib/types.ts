// Hand-written mirrors of backend/models/prediction.py — keep in sync.
export interface ComponentInfo {
  key: string;
  name: string;
  english_name: string;
  category: string;
  summary: string;
  function: string;
  usage_areas: string[];
  identify_tips: string[];
  typical_values: string;
  packages: string[];
  how_it_works: string | null; // only microcontroller boards
  specs: string[];
}

export interface ResistorValue {
  ohms: number;
  formatted: string;
  tolerance_percent: number;
  min_ohms: number;
  max_ohms: number;
  formula: string;
}

export interface ResistorReading {
  bands: string[];
  detected: boolean;
  note: string | null;
  value: ResistorValue | null;
}

export interface ClassProb {
  key: string;
  name: string;
  probability: number;
}

export interface Prediction {
  id: string;
  created_at: string;
  label: string;
  name: string;
  confidence: number;
  is_component: boolean;
  source: "cnn" | "cnn+ai";
  cnn_label: string;
  cnn_confidence: number;
  cnn_top3: ClassProb[];
  ai_used: boolean;
  ai_label: string | null;
  ai_confidence: number | null;
  ai_note: string | null;
  agreement: boolean | null;
  thumbnail: string;
  info: ComponentInfo | null;
  resistor: ResistorReading | null;
  feedback_label?: string | null;
}

export interface EpochRow {
  epoch: number;
  train_loss: number;
  train_acc: number;
  val_loss: number;
  val_acc: number;
}

export interface ClassMetric {
  key: string;
  name: string;
  precision: number;
  recall: number;
  f1: number;
  support: number;
  total_images: number;
}

export interface ModelInfo {
  available: boolean;
  arch: string;
  architecture: string;
  device?: string | null;
  augmentation?: string | null;
  framework: string;
  input_size: number;
  epochs: number;
  batch_size: number;
  optimizer: string;
  loss: string;
  train_size: number;
  val_size: number;
  best_val_acc: number;
  classes: string[];
  class_names: string[];
  history: EpochRow[];
  confusion_matrix: number[][];
  per_class: ClassMetric[];
  trained_at: string;
  ai_policy: string;
  ai_model: string;
}

export interface FeedbackResult {
  total: number;
  per_class: Record<string, number>;
}

export interface LivePrediction {
  label: string;
  name: string;
  confidence: number;
  top3: ClassProb[];
  probs: ClassProb[]; // all classes, best first
}

export interface Sample {
  key: string;
  name: string;
  url: string;
}
