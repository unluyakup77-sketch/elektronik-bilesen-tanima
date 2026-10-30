import { useQuery } from "@tanstack/react-query";
import { Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Layout, SectionLabel } from "@/components/Layout";
import { apiGet } from "@/lib/api";
import type { ModelInfo } from "@/lib/types";

// Gövde katmanları mimariye göre değişir (backend/ml/train.py → build_model)
const BACKBONES: Record<string, { t: string; d: string; features: number }> = {
  efficientnet_b0: { t: "EfficientNet-B0 gövde", d: "MBConv blokları (derinlemesine ayrılabilir evrişim + squeeze-excitation), ImageNet ile ön-eğitimli", features: 1280 },
  efficientnet_v2_s: { t: "EfficientNetV2-S gövde", d: "Fused-MBConv + MBConv blokları, ImageNet ile ön-eğitimli", features: 1280 },
  convnext_tiny: { t: "ConvNeXt-Tiny gövde", d: "4 aşamalı modern CNN: 7×7 derinlemesine evrişim blokları, LayerNorm, GELU; ImageNet ile ön-eğitimli", features: 768 },
};

function pipeline(m: ModelInfo) {
  const b = BACKBONES[m.arch] ?? BACKBONES.efficientnet_b0;
  const n = m.classes.length;
  return [
    { t: "Giriş", d: `RGB görsel → ${m.input_size}×${m.input_size}, ImageNet normalizasyonu` },
    { t: b.t, d: b.d },
    { t: "Global Average Pooling", d: `${b.features} boyutlu öznitelik vektörü` },
    { t: "Dropout 0.3 + Dense", d: `${b.features} → ${n} sınıf, softmax` },
    { t: "Karar", d: "En yüksek softmax olasılığına sahip sınıf (argmax); \"Bileşen değil\" sınıfı boş kadrajı ayırır" },
  ];
}

const tooltipStyle = { background: "#111827", border: "1px solid #1F2937", borderRadius: 8, color: "#F1F5F9" };

function Stat({ label, value, testId }: { label: string; value: string; testId: string }) {
  return (
    <div className="rounded-xl border border-slate-800 bg-[#111827]/90 p-5">
      <SectionLabel>{label}</SectionLabel>
      <p className="mt-2 font-mono text-3xl font-semibold text-slate-50" data-testid={testId}>{value}</p>
    </div>
  );
}

function ConfusionMatrix({ m }: { m: ModelInfo }) {
  const n = m.classes.length;
  return (
    <div className="overflow-x-auto" data-testid="model-confusion-matrix">
      <div className="inline-grid gap-[2px] min-w-max" style={{ gridTemplateColumns: `130px repeat(${n}, 30px)` }}>
        <div />
        {m.class_names.map((c) => (
          <div key={c} className="h-28 flex items-end justify-center">
            <span className="font-mono text-[10px] text-slate-400 whitespace-nowrap [writing-mode:vertical-rl] rotate-180">{c}</span>
          </div>
        ))}
        {m.confusion_matrix.map((row, i) => {
          const total = row.reduce((a, b) => a + b, 0) || 1;
          return [
            <div key={`l${i}`} className="font-mono text-[10px] text-slate-400 flex items-center justify-end pr-2 truncate">{m.class_names[i]}</div>,
            ...row.map((v, j) => {
              const r = v / total;
              const bg = i === j ? `rgba(0,240,255,${0.12 + r * 0.8})` : v ? `rgba(239,68,68,${0.15 + r * 1.5})` : "#0d1424";
              return (
                <div key={`${i}-${j}`} title={`Gerçek: ${m.class_names[i]} · Tahmin: ${m.class_names[j]} · ${v}`} className="h-[30px] w-[30px] grid place-items-center rounded-[3px] font-mono text-[10px] text-slate-100" style={{ background: bg }}>
                  {v || ""}
                </div>
              );
            }),
          ];
        })}
      </div>
      <p className="mt-3 text-xs text-slate-500">Satırlar gerçek sınıf, sütunlar modelin tahmini. Köşegen (camgöbeği) doğru tahminleri, kırmızı hücreler karışan sınıfları gösterir.</p>
    </div>
  );
}

export default function ModelLab() {
  const { data, isError, isLoading } = useQuery({ queryKey: ["model"], queryFn: () => apiGet<ModelInfo>("/model") });
  const m = isError ? undefined : data;
  const history = (m?.history ?? []).map((h) => ({ ...h, train_acc: +(h.train_acc * 100).toFixed(1), val_acc: +(h.val_acc * 100).toFixed(1) }));
  const dist = (m?.per_class ?? []).map((p) => ({ name: p.name, adet: p.total_images }));
  const macroF1 = m ? m.per_class.reduce((a, p) => a + p.f1, 0) / m.per_class.length : 0;

  return (
    <Layout>
      <section className="max-w-3xl">
        <SectionLabel>Derin öğrenme laboratuvarı</SectionLabel>
        <h1 className="mt-3 text-4xl sm:text-5xl font-bold tracking-tight text-slate-50">Model & Metrikler</h1>
        <p className="mt-3 text-slate-400 leading-relaxed">
          Bileşen tanıma, PyTorch ile eğitilmiş bir Evrişimli Sinir Ağı (CNN) ile yapılır. Aşağıda mimari, veri seti ve doğrulama sonuçları yer alır.
        </p>
      </section>

      {isLoading && <div className="mt-8 h-32 rounded-xl border border-slate-800 bg-[#111827] animate-pulse" />}
      {(isError || (!isLoading && !m)) && <p className="mt-8 text-sm text-slate-500" data-testid="model-unavailable">Model metrikleri şu an yüklenemedi.</p>}

      {m && (
        <>
          <div className="mt-8 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <Stat label="Doğrulama doğruluğu" value={`%${(m.best_val_acc * 100).toFixed(1)}`} testId="model-val-accuracy" />
            <Stat label="Makro F1" value={macroF1.toFixed(3)} testId="model-macro-f1" />
            <Stat label="Görsel sayısı" value={`${m.train_size + m.val_size}`} testId="model-dataset-size" />
            <Stat label="Sınıf sayısı" value={`${m.classes.length}`} testId="model-class-count" />
          </div>

          <div className="mt-6 grid grid-cols-1 lg:grid-cols-12 gap-6">
            <div className="lg:col-span-5 rounded-xl border border-slate-800 bg-[#111827]/90 p-6">
              <SectionLabel>Mimari · {m.framework}</SectionLabel>
              <ol className="mt-4 relative border-l border-cyan-400/30 ml-2 space-y-4" data-testid="model-pipeline">
                {pipeline(m).map((p) => (
                  <li key={p.t} className="pl-5 relative">
                    <span className="absolute -left-[5px] top-1.5 h-2.5 w-2.5 rounded-full bg-cyan-400 shadow-[0_0_8px_rgba(0,240,255,0.8)]" />
                    <p className="font-heading font-semibold text-slate-100">{p.t}</p>
                    <p className="text-sm text-slate-400">{p.d}</p>
                  </li>
                ))}
              </ol>
              <dl className="mt-6 grid grid-cols-2 gap-x-4 gap-y-2 font-mono text-xs">
                <dt className="text-slate-500">Model</dt><dd className="text-slate-300">{m.architecture}</dd>
                <dt className="text-slate-500">Epoch</dt><dd className="text-slate-300">{m.epochs}</dd>
                <dt className="text-slate-500">Batch</dt><dd className="text-slate-300">{m.batch_size}</dd>
                <dt className="text-slate-500">Optimizer</dt><dd className="text-slate-300">{m.optimizer}</dd>
                <dt className="text-slate-500">Kayıp</dt><dd className="text-slate-300">{m.loss}</dd>
                {m.augmentation && <><dt className="text-slate-500">Veri artırma</dt><dd className="text-slate-300">{m.augmentation}</dd></>}
                {m.device && <><dt className="text-slate-500">Eğitim donanımı</dt><dd className="text-slate-300">{m.device}</dd></>}
                <dt className="text-slate-500">Eğitim / Doğr.</dt><dd className="text-slate-300">{m.train_size} / {m.val_size} (%80 / %20)</dd>
              </dl>
            </div>

            <div className="lg:col-span-7 grid gap-6">
              <div className="rounded-xl border border-slate-800 bg-[#111827]/90 p-6">
                <SectionLabel>Kayıp (loss) · epoch</SectionLabel>
                <div className="mt-4 h-56" data-testid="model-loss-chart">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={history}>
                      <CartesianGrid stroke="#1F2937" strokeDasharray="3 3" />
                      <XAxis dataKey="epoch" stroke="#64748B" fontSize={11} />
                      <YAxis stroke="#64748B" fontSize={11} />
                      <Tooltip contentStyle={tooltipStyle} />
                      <Legend wrapperStyle={{ fontSize: 12 }} />
                      <Line type="monotone" dataKey="train_loss" name="Eğitim" stroke="#00F0FF" strokeWidth={2} dot={false} />
                      <Line type="monotone" dataKey="val_loss" name="Doğrulama" stroke="#C084FC" strokeWidth={2} dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
              <div className="rounded-xl border border-slate-800 bg-[#111827]/90 p-6">
                <SectionLabel>Doğruluk (%) · epoch</SectionLabel>
                <div className="mt-4 h-56" data-testid="model-accuracy-chart">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={history}>
                      <CartesianGrid stroke="#1F2937" strokeDasharray="3 3" />
                      <XAxis dataKey="epoch" stroke="#64748B" fontSize={11} />
                      <YAxis stroke="#64748B" fontSize={11} domain={[0, 100]} />
                      <Tooltip contentStyle={tooltipStyle} />
                      <Legend wrapperStyle={{ fontSize: 12 }} />
                      <Line type="monotone" dataKey="train_acc" name="Eğitim" stroke="#00F0FF" strokeWidth={2} dot={false} />
                      <Line type="monotone" dataKey="val_acc" name="Doğrulama" stroke="#10B981" strokeWidth={2} dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          </div>

          <div className="mt-6 rounded-xl border border-slate-800 bg-[#111827]/90 p-6">
            <SectionLabel>Karışıklık matrisi (doğrulama seti)</SectionLabel>
            <div className="mt-4"><ConfusionMatrix m={m} /></div>
          </div>

          <div className="mt-6 grid grid-cols-1 lg:grid-cols-12 gap-6">
            <div className="lg:col-span-7 rounded-xl border border-slate-800 bg-[#111827]/90 p-6">
              <SectionLabel>Sınıf bazlı metrikler</SectionLabel>
              <div className="mt-4" data-testid="model-per-class-table">
                <Table>
                  <TableHeader>
                    <TableRow className="border-slate-800">
                      <TableHead>Sınıf</TableHead><TableHead className="text-right">Precision</TableHead><TableHead className="text-right">Recall</TableHead><TableHead className="text-right">F1</TableHead><TableHead className="text-right">Örnek</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {m.per_class.map((p) => (
                      <TableRow key={p.key} className="border-slate-800 font-mono text-xs">
                        <TableCell className="font-sans text-sm text-slate-200">{p.name}</TableCell>
                        <TableCell className="text-right">{p.precision.toFixed(2)}</TableCell>
                        <TableCell className="text-right">{p.recall.toFixed(2)}</TableCell>
                        <TableCell className="text-right text-cyan-300">{p.f1.toFixed(2)}</TableCell>
                        <TableCell className="text-right text-slate-400">{p.support}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            </div>
            <div className="lg:col-span-5 rounded-xl border border-slate-800 bg-[#111827]/90 p-6">
              <SectionLabel>Veri seti dağılımı</SectionLabel>
              <div className="mt-4 h-[480px]" data-testid="model-dataset-chart">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={dist} layout="vertical" margin={{ left: 20 }}>
                    <CartesianGrid stroke="#1F2937" strokeDasharray="3 3" horizontal={false} />
                    <XAxis type="number" stroke="#64748B" fontSize={11} />
                    <YAxis type="category" dataKey="name" stroke="#64748B" fontSize={10} width={110} />
                    <Tooltip contentStyle={tooltipStyle} cursor={{ fill: "rgba(0,240,255,0.05)" }} />
                    <Bar dataKey="adet" name="Görsel" fill="#00F0FF" radius={[0, 4, 4, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <p className="mt-3 text-xs text-slate-500 leading-relaxed">
                Kaynaklar: Kaggle "electronic-components", HuggingFace "qipchip31/electronic_components" ve web görsel araması. Görseller CLIP ile otomatik temizlendi, tekrar edenler silindi.
              </p>
            </div>
          </div>
        </>
      )}
    </Layout>
  );
}
