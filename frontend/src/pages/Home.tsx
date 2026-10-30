import { useRef, useState, type ChangeEvent, type DragEvent } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { AnimatePresence, motion } from "motion/react";
import { Camera, ImageUp, Loader2, ScanLine, ShieldCheck, Trash2, Upload, Video, X } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Layout, SectionLabel } from "@/components/Layout";
import { ScanResult } from "@/components/ScanResult";
import { LiveCamera } from "@/components/LiveCamera";
import { apiDelete, apiGet, apiUpload, errorMessage } from "@/lib/api";
import type { Prediction, Sample } from "@/lib/types";
import { cn } from "@/lib/utils";

const ACCEPT = "image/jpeg,image/png,image/webp,image/heic,image/heif";

const STEPS = [
  { n: "01", t: "Fotoğraf yükle", d: "Bileşeni düz bir zemine koyup yakından çek veya galeriden seç." },
  { n: "02", t: "CNN tarar", d: "ConvNeXt-Tiny tabanlı CNN, 19 bileşen sınıfı ve \"bileşen değil\" için olasılık hesaplar; kararı CNN verir." },
  { n: "03", t: "Canlı kamera", d: "Gerçek parçayı kameraya tut; CNN saniyede yaklaşık bir kez tanır." },
  { n: "04", t: "Sonuç + bilgi", d: "Bileşen adı, güven yüzdesi, ne işe yaradığı ve devre sembolü gösterilir." },
];

export default function Home() {
  const qc = useQueryClient();
  const fileInput = useRef<HTMLInputElement>(null);
  const cameraInput = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const [doubleCheck, setDoubleCheck] = useState(false);
  const [result, setResult] = useState<Prediction | null>(null);
  const [liveOpen, setLiveOpen] = useState(false);
  const [advancedOpen, setAdvancedOpen] = useState(false);

  const samples = useQuery({ queryKey: ["samples"], queryFn: () => apiGet<Sample[]>("/samples") });
  const history = useQuery({ queryKey: ["predictions"], queryFn: () => apiGet<Prediction[]>("/predictions") });

  const scan = useMutation({
    mutationFn: (f: File) => {
      const form = new FormData();
      form.append("file", f);
      form.append("double_check", String(doubleCheck));
      return apiUpload<Prediction>("/predict", form);
    },
    onSuccess: (p) => {
      setResult(p);
      qc.invalidateQueries({ queryKey: ["predictions"] });
      toast.success(p.is_component ? `${p.name} tespit edildi` : "Bileşen tanımlanamadı");
    },
    onError: (e) => toast.error(errorMessage(e, "Tarama başarısız oldu. Tekrar deneyin.")),
  });

  const remove = useMutation({
    mutationFn: (id: string) => apiDelete<{ deleted: string }>(`/predictions/${id}`),
    onSuccess: (_d, id) => {
      if (result?.id === id) setResult(null);
      qc.invalidateQueries({ queryKey: ["predictions"] });
    },
    onError: () => toast.error("Kayıt silinemedi."),
  });

  function pick(f: File | undefined | null) {
    if (!f) return;
    if (!f.type.startsWith("image/")) {
      toast.error("Lütfen bir görsel dosyası seçin (JPG, PNG, WEBP).");
      return;
    }
    if (preview) URL.revokeObjectURL(preview);
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setResult(null);
  }

  function onInput(e: ChangeEvent<HTMLInputElement>) {
    pick(e.target.files?.[0]);
    e.target.value = "";
  }

  function onDrop(e: DragEvent<HTMLDivElement>) {
    e.preventDefault();
    setDragging(false);
    pick(e.dataTransfer.files?.[0]);
  }

  async function pickSample(s: Sample) {
    try {
      const blob = await (await fetch(s.url)).blob();
      pick(new File([blob], `${s.key}.jpg`, { type: "image/jpeg" }));
    } catch {
      toast.error("Örnek görsel yüklenemedi.");
    }
  }

  function onLiveCapture(f: File) {
    pick(f);
    scan.mutate(f);
  }

  function clear() {
    if (preview) URL.revokeObjectURL(preview);
    setFile(null);
    setPreview(null);
    setResult(null);
  }

  const scanning = scan.isPending;
  const historyItems = history.isError ? [] : (history.data ?? []);

  return (
    <Layout>
      <section className="mb-10 max-w-3xl">
        <SectionLabel>Elektronik bileşen tanıma · CNN</SectionLabel>
        <h1 className="mt-3 text-4xl sm:text-5xl lg:text-[56px] font-bold tracking-tight text-slate-50 leading-[1.05]">
          Bilmediğin bileşenin fotoğrafını çek, <span className="text-cyan-300">ne olduğunu</span> ve <span className="text-cyan-300">ne işe yaradığını</span> öğren.
        </h1>
        <p className="mt-4 text-slate-400 leading-relaxed text-base sm:text-lg">
          Direnç, LED, kondansatör, transistör, entegre, Arduino, ESP32, STM32 ve daha fazlası (19 sınıf). Kararı kendi eğittiğimiz CNN modeli verir; dirençlerin renk kodundan ohm değeri hesaplanır.
        </p>
      </section>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        {/* Sol: yükleme */}
        <div className="lg:col-span-5 flex flex-col gap-5 lg:sticky lg:top-24">
          <div
            data-testid="upload-dropzone"
            onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
            onClick={() => !preview && fileInput.current?.click()}
            className={cn(
              "relative aspect-[4/3] rounded-xl border-2 border-dashed overflow-hidden bg-[#111827]/80 transition-colors duration-200",
              dragging ? "border-cyan-400 bg-cyan-400/5" : "border-slate-700 hover:border-cyan-500/50",
              !preview && "cursor-pointer",
            )}
          >
            {preview ? (
              <>
                <img src={preview} alt="Yüklenen bileşen" className="h-full w-full object-contain bg-black/40" data-testid="upload-preview-image" />
                {scanning && (
                  <div className="absolute inset-0 pointer-events-none" data-testid="scan-animation">
                    <div className="absolute inset-0 bg-cyan-400/5" />
                    <div className="absolute left-0 right-0 h-[3px] bg-cyan-300 shadow-[0_0_18px_4px_rgba(0,240,255,0.7)] animate-scan-beam" />
                    <div className="absolute inset-3 border border-cyan-400/40 rounded-lg" />
                  </div>
                )}
                {!scanning && (
                  <button
                    type="button"
                    onClick={(e) => { e.stopPropagation(); clear(); }}
                    className="absolute top-3 right-3 grid place-items-center h-9 w-9 rounded-md bg-black/60 text-slate-200 hover:bg-black/80 transition-colors"
                    aria-label="Görseli kaldır"
                    data-testid="upload-clear-button"
                  >
                    <X className="h-4 w-4" />
                  </button>
                )}
              </>
            ) : (
              <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 p-6 text-center">
                <span className="grid place-items-center h-14 w-14 rounded-xl border border-cyan-400/30 bg-cyan-400/10 text-cyan-300">
                  <ImageUp className="h-7 w-7" />
                </span>
                <p className="font-heading text-lg font-semibold text-slate-100">Fotoğrafı buraya sürükle</p>
                <p className="text-sm text-slate-400">veya tıklayıp seç · JPG, PNG, WEBP · en fazla 10 MB</p>
              </div>
            )}
          </div>

          <input ref={fileInput} type="file" accept={ACCEPT} className="hidden" onChange={onInput} data-testid="upload-file-input" />
          <input ref={cameraInput} type="file" accept="image/*" capture="environment" className="hidden" onChange={onInput} data-testid="upload-camera-input" />

          <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
            <Button variant="outline" className="h-11" onClick={() => fileInput.current?.click()} disabled={scanning} data-testid="upload-select-button">
              <Upload className="h-4 w-4" /> Fotoğraf Seç
            </Button>
            <Button variant="outline" className="h-11" onClick={() => cameraInput.current?.click()} disabled={scanning} data-testid="upload-camera-button">
              <Camera className="h-4 w-4" /> Fotoğraf Çek
            </Button>
            <Button className="h-11 col-span-2 sm:col-span-1 bg-cyan-400/15 text-cyan-200 border border-cyan-400/40 hover:bg-cyan-400/25" onClick={() => setLiveOpen(true)} disabled={scanning} data-testid="live-camera-open-button">
              <Video className="h-4 w-4" /> Canlı Kamera
            </Button>
          </div>
          <LiveCamera open={liveOpen} onOpenChange={setLiveOpen} onCapture={onLiveCapture} scanning={scanning} lastResult={result} />

          <button
            type="button"
            onClick={() => setAdvancedOpen((v) => !v)}
            aria-expanded={advancedOpen}
            className="self-start text-xs text-slate-500 hover:text-slate-300 transition-colors"
            data-testid="advanced-settings-toggle"
          >
            {advancedOpen ? "▾" : "▸"} Gelişmiş ayarlar
          </button>
          {advancedOpen && (
          <button
            type="button"
            role="switch"
            aria-checked={doubleCheck}
            onClick={() => setDoubleCheck((v) => !v)}
            data-testid="double-check-toggle"
            className="flex items-center gap-3 rounded-lg border border-slate-800 bg-[#111827]/80 p-3 text-left transition-colors hover:border-purple-500/40"
          >
            <span className={cn("relative h-6 w-11 rounded-full transition-colors duration-200 shrink-0", doubleCheck ? "bg-purple-500" : "bg-slate-700")}>
              <span className={cn("absolute top-0.5 h-5 w-5 rounded-full bg-white transition-transform duration-200", doubleCheck ? "translate-x-[22px]" : "translate-x-0.5")} />
            </span>
            <span className="min-w-0">
              <span className="flex items-center gap-1.5 text-sm font-medium text-slate-100"><ShieldCheck className="h-4 w-4 text-purple-300" /> Çift kontrol (her zaman AI doğrulasın)</span>
              <span className="block text-xs text-slate-400">Kapalıyken karar tamamen CNN modelinindir; açınca Gemini ikinci kontrol yapar.</span>
            </span>
          </button>
          )}

          <Button
            size="lg"
            className="h-12 text-base font-semibold shadow-[0_0_24px_rgba(0,240,255,0.25)] active:scale-[0.99] transition-transform"
            disabled={!file || scanning}
            onClick={() => file && scan.mutate(file)}
            data-testid="scan-submit-button"
          >
            {scanning ? <><Loader2 className="h-5 w-5 animate-spin" /> Taranıyor…</> : <><ScanLine className="h-5 w-5" /> Bileşeni Tara</>}
          </Button>

          <div>
            <SectionLabel>Örnek görsellerle dene</SectionLabel>
            <div className="mt-3 flex gap-2 overflow-x-auto pb-1" data-testid="sample-list">
              {(samples.isError ? [] : (samples.data ?? [])).map((s) => (
                <button
                  key={s.key}
                  type="button"
                  onClick={() => pickSample(s)}
                  disabled={scanning}
                  data-testid={`sample-item-${s.key}`}
                  className="group shrink-0 w-20 rounded-lg border border-slate-800 bg-[#111827] overflow-hidden hover:border-cyan-500/50 transition-colors"
                  title={s.name}
                >
                  <img src={s.url} alt={s.name} className="h-16 w-full object-cover" loading="lazy" />
                  <span className="block truncate px-1 py-1 text-[10px] text-slate-400 group-hover:text-slate-200">{s.name}</span>
                </button>
              ))}
              {samples.isLoading && <span className="text-xs text-slate-500">Yükleniyor…</span>}
            </div>
          </div>
        </div>

        {/* Sağ: sonuç */}
        <div className="lg:col-span-7 flex flex-col gap-6" aria-live="polite">
          <AnimatePresence mode="wait">
            {result ? (
              <ScanResult key={result.id} result={result} />
            ) : (
              <motion.div
                key="empty"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="rounded-xl border border-slate-800 bg-[#111827]/70 p-6 sm:p-8"
                data-testid="result-empty-state"
              >
                <SectionLabel>{scanning ? "Analiz ediliyor" : "Nasıl çalışır?"}</SectionLabel>
                {scanning ? (
                  <div className="mt-6 flex flex-col gap-3 font-mono text-sm text-slate-400">
                    <span>&gt; Görsel 224×224 boyutuna getiriliyor…</span>
                    <span>&gt; ConvNeXt-Tiny ileri yayılım (forward pass)…</span>
                    <span>&gt; Softmax olasılıkları hesaplanıyor…</span>
                    <span className="text-purple-300">&gt; {doubleCheck ? "AI çift kontrolü yapılıyor" : "Sonuç hazırlanıyor"}<span className="animate-blink">_</span></span>
                  </div>
                ) : (
                  <ol className="mt-5 grid grid-cols-1 sm:grid-cols-2 gap-4">
                    {STEPS.map((s) => (
                      <li key={s.n} className="rounded-lg border border-slate-800 bg-[#0d1424] p-4">
                        <span className="font-mono text-xs text-cyan-300">{s.n}</span>
                        <p className="mt-1 font-heading font-semibold text-slate-100">{s.t}</p>
                        <p className="mt-1 text-sm text-slate-400 leading-relaxed">{s.d}</p>
                      </li>
                    ))}
                  </ol>
                )}
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>

      {/* Geçmiş */}
      <section className="mt-14" data-testid="history-section">
        <div className="flex items-end justify-between">
          <div>
            <SectionLabel>Geçmiş taramalar</SectionLabel>
            <h2 className="mt-1 text-2xl font-bold tracking-tight text-slate-100">Son taranan bileşenler</h2>
          </div>
          <span className="font-mono text-xs text-slate-500">{historyItems.length} kayıt</span>
        </div>
        {historyItems.length === 0 ? (
          <p className="mt-4 text-sm text-slate-500" data-testid="history-empty">Henüz tarama yok. İlk bileşenini yukarıdan tara.</p>
        ) : (
          <ul className="mt-5 grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-4" data-testid="history-list">
            {historyItems.map((h) => (
              <li key={h.id} className="group relative rounded-lg border border-slate-800 bg-[#111827] overflow-hidden hover:border-cyan-500/40 transition-colors" data-testid={`history-item-${h.id}`}>
                <button type="button" className="block w-full text-left" onClick={() => { setResult(h); setPreview(h.thumbnail); setFile(null); window.scrollTo({ top: 0, behavior: "smooth" }); }} data-testid={`history-open-${h.id}`}>
                  <img src={h.thumbnail} alt={h.name} className="h-28 w-full object-cover" />
                  <span className="block px-2.5 pt-2 text-sm font-medium text-slate-100 truncate">{h.name}</span>
                  <span className="flex items-center justify-between px-2.5 pb-2.5 font-mono text-[11px] text-slate-400">
                    <span>%{(h.confidence * 100).toFixed(0)}</span>
                    <span className={h.source === "cnn" ? "text-sky-300" : "text-purple-300"}>{h.source === "cnn" ? "CNN" : "CNN+AI"}</span>
                  </span>
                </button>
                <button
                  type="button"
                  onClick={() => remove.mutate(h.id)}
                  className="absolute top-2 right-2 grid place-items-center h-8 w-8 rounded-md bg-black/70 text-slate-300 opacity-100 sm:opacity-0 sm:group-hover:opacity-100 hover:text-red-300 transition-opacity"
                  aria-label="Kaydı sil"
                  data-testid={`history-delete-${h.id}`}
                >
                  <Trash2 className="h-4 w-4" />
                </button>
              </li>
            ))}
          </ul>
        )}
      </section>
    </Layout>
  );
}
