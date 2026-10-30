import { useCallback, useEffect, useRef, useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { Camera, Loader2, Pause, Play, RefreshCw, ScanLine, VideoOff } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { apiUpload, errorMessage } from "@/lib/api";
import type { ClassProb, LivePrediction, Prediction } from "@/lib/types";

const LIVE_INTERVAL_MS = 500; // önceki istek bitmeden yenisi gönderilmez; fiili hız sunucu süresiyle sınırlı (~0,5 sn)
const SMOOTH_FRAMES = 3; // son N karenin olasılıkları ortalanır (tek karelik titremeleri bastırır)
const SWITCH_CONF = 0.5; // yeni kare farklı bir sınıfı bu güvenle görürse ortalama sıfırlanır (hızlı geçiş)
const LIVE_SURE = 0.55; // ortalama güven bunun üstündeyse "kesin", altındaysa "muhtemelen"
const LIVE_MAYBE = 0.25;
const UNKNOWN = "bilinmiyor";

// Son birkaç karenin sınıf olasılıklarının ortalaması, en yüksekten düşüğe
function averageFrames(frames: LivePrediction[]): ClassProb[] {
  const sum = new Map<string, ClassProb>();
  for (const f of frames) {
    for (const p of f.probs.length ? f.probs : f.top3) {
      const prev = sum.get(p.key);
      sum.set(p.key, { ...p, probability: (prev?.probability ?? 0) + p.probability / frames.length });
    }
  }
  return [...sum.values()].sort((a, b) => b.probability - a.probability);
}

interface Props {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onCapture: (file: File) => void;
  scanning: boolean;
  lastResult: Prediction | null;
}

export function LiveCamera({ open, onOpenChange, onCapture, scanning, lastResult }: Props) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [facing, setFacing] = useState<"environment" | "user">("environment");
  const [error, setError] = useState<string | null>(null);
  const [ready, setReady] = useState(false);
  const [captured, setCaptured] = useState(false);
  const [auto, setAuto] = useState(true);
  const [frames, setFrames] = useState<LivePrediction[]>([]);
  const [liveError, setLiveError] = useState<string | null>(null);
  const [stream, setStream] = useState<MediaStream | null>(null);

  const liveScan = useMutation({
    mutationFn: (f: File) => {
      const form = new FormData();
      form.append("file", f);
      return apiUpload<LivePrediction>("/predict/live", form);
    },
    onSuccess: (p) => {
      setLiveError(null);
      setFrames((fs) => {
        // Kadraja başka bir parça girdiyse (yeni kare farklı bir sınıfı net görüyorsa) eski kareleri at:
        // ortalama, parça değişince sonucun saniyelerce eski parçada takılı kalmasına yol açıyordu.
        const prevTop = averageFrames(fs)[0]?.key;
        if (prevTop && p.label !== prevTop && p.confidence >= SWITCH_CONF) return [p];
        return [...fs, p].slice(-SMOOTH_FRAMES);
      });
    },
    onError: (e) => setLiveError(errorMessage(e, "Sunucuya ulaşılamadı. Backend penceresi açık mı?")),
  });

  // Akışı video öğesine bağla. Kamera, diyalog içeriği ekrana gelmeden açılabildiği için
  // bağlama hem akış hazır olduğunda hem de video öğesi DOM'a girdiğinde denenir.
  const attach = useCallback((video: HTMLVideoElement | null, s: MediaStream | null) => {
    if (!video || !s || video.srcObject === s) return;
    video.srcObject = s;
    video.onloadedmetadata = () => {
      video.play().catch(() => undefined);
      setReady(true);
    };
  }, []);
  const videoCallbackRef = useCallback(
    (el: HTMLVideoElement | null) => {
      videoRef.current = el;
      attach(el, stream);
    },
    [attach, stream],
  );
  useEffect(() => attach(videoRef.current, stream), [attach, stream]);

  useEffect(() => {
    if (!open) return;
    let s: MediaStream | null = null;
    let cancelled = false;
    setError(null);
    setReady(false);
    if (!navigator.mediaDevices?.getUserMedia) {
      setError("Bu tarayıcı kamera erişimini desteklemiyor (kamera için siteyi http://localhost:3000 adresinden açın).");
      return;
    }
    navigator.mediaDevices
      .getUserMedia({ video: { facingMode: { ideal: facing }, width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false })
      .then((ms) => {
        if (cancelled) {
          ms.getTracks().forEach((t) => t.stop());
          return;
        }
        s = ms;
        setStream(ms);
      })
      .catch((e: unknown) => {
        const name = e instanceof DOMException ? e.name : "";
        setError(
          name === "NotAllowedError"
            ? "Kamera izni verilmedi. Tarayıcının adres çubuğundan kamera iznini açıp tekrar dene."
            : name === "NotFoundError"
              ? "Bu cihazda kamera bulunamadı."
              : "Kamera açılamadı. Başka bir uygulama kamerayı kullanıyor olabilir.",
        );
      });
    return () => {
      cancelled = true;
      s?.getTracks().forEach((t) => t.stop());
      setStream(null);
    };
  }, [open, facing]);

  useEffect(() => {
    if (!open) {
      setCaptured(false);
      setFrames([]);
      setLiveError(null);
    }
  }, [open]);

  // Hedef çerçevesinin içini kırpar: CNN tek ve ortalanmış bileşenle eğitildiği için daha doğru sonuç verir.
  function grabFrame(): Promise<File | null> {
    const video = videoRef.current;
    if (!video || !video.videoWidth) return Promise.resolve(null);
    const vw = video.videoWidth;
    const vh = video.videoHeight;
    const side = Math.round(Math.min(vw, vh) * 0.8);
    const canvas = document.createElement("canvas");
    canvas.width = side;
    canvas.height = side;
    canvas.getContext("2d")?.drawImage(video, (vw - side) / 2, (vh - side) / 2, side, side, 0, 0, side, side);
    return new Promise((resolve) =>
      canvas.toBlob((blob) => resolve(blob ? new File([blob], `kamera-${Date.now()}.jpg`, { type: "image/jpeg" }) : null), "image/jpeg", 0.9),
    );
  }

  // Otomatik canlı tanıma: ~1.2 sn'de bir kare CNN'e gönderilir (önceki istek bitmeden yenisi gönderilmez).
  const busy = liveScan.isPending;
  useEffect(() => {
    if (!open || !auto || !ready || error || scanning) return;
    const id = window.setInterval(async () => {
      if (busy) return;
      const f = await grabFrame();
      if (f) liveScan.mutate(f);
    }, LIVE_INTERVAL_MS);
    return () => window.clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, auto, ready, error, scanning, busy]);

  async function capture() {
    const f = await grabFrame();
    if (!f) return;
    setCaptured(true);
    onCapture(f);
  }

  const showResult = captured && !scanning && lastResult;
  const avg = averageFrames(frames);
  const showLive = auto && avg.length > 0;
  // En olası BİLEŞEN ("bilinmiyor" hariç); kadraj boşsa "bilinmiyor" baskın gelir
  const best = avg.find((p) => p.key !== UNKNOWN) ?? null;
  const emptyProb = avg.find((p) => p.key === UNKNOWN)?.probability ?? 0;
  const status: "sure" | "maybe" | "empty" =
    best && best.probability >= LIVE_SURE ? "sure" : best && best.probability >= LIVE_MAYBE && best.probability > emptyProb * 0.5 ? "maybe" : "empty";
  const others = avg.filter((p) => p.key !== UNKNOWN && p.key !== best?.key).slice(0, 2);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-3xl bg-[#111827] border-slate-800" data-testid="live-camera-dialog">
        <DialogHeader>
          <DialogTitle className="font-heading text-xl">Canlı Kamera ile Tara</DialogTitle>
          <DialogDescription className="text-slate-400">
            Gerçek parçayı çerçevenin içine, kameraya yakın tut. Canlı tanıma açıkken CNN her ~1 saniyede bir tahmin yapar; detaylı sonuç (renk kodu, açıklama) için "Kareyi Tara"ya bas.
          </DialogDescription>
        </DialogHeader>

        <div className="relative aspect-video rounded-lg overflow-hidden bg-black border border-slate-800">
          {error ? (
            <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 p-6 text-center text-slate-300" data-testid="live-camera-error">
              <VideoOff className="h-8 w-8 text-red-300" />
              <p className="text-sm">{error}</p>
            </div>
          ) : (
            <>
              <video ref={videoCallbackRef} autoPlay playsInline muted className="h-full w-full object-cover" data-testid="live-camera-video" />
              {!ready && (
                <div className="absolute inset-0 grid place-items-center text-slate-400"><Loader2 className="h-6 w-6 animate-spin" /></div>
              )}
              {/* hedef çerçevesi */}
              <div className="pointer-events-none absolute inset-y-[10%] left-1/2 -translate-x-1/2 aspect-square border-2 border-cyan-300/60 rounded-lg shadow-[0_0_0_9999px_rgba(0,0,0,0.25)]" />
              {showLive && (
                <div className="absolute left-3 top-3 right-3 flex items-start justify-between gap-3" data-testid="live-camera-live-result">
                  <div className="rounded-lg bg-black/75 backdrop-blur px-3 py-2 min-w-0">
                    <p className="font-mono text-[10px] uppercase tracking-[0.18em] text-cyan-300/80 flex items-center gap-1.5">
                      <span className="h-1.5 w-1.5 rounded-full bg-red-500 animate-pulse" /> Canlı CNN
                    </p>
                    {status === "sure" && best && (
                      <p className="font-heading text-xl font-semibold text-slate-50 truncate" data-testid="live-camera-live-name">{best.name}</p>
                    )}
                    {status === "maybe" && best && (
                      <p className="font-heading text-lg font-semibold text-amber-200 truncate" data-testid="live-camera-live-name">
                        Muhtemelen: {best.name}
                      </p>
                    )}
                    {status === "empty" && (
                      <p className="text-sm text-slate-300" data-testid="live-camera-live-searching">Bileşen görünmüyor… parçayı çerçevenin ortasına, kameraya yaklaştır</p>
                    )}
                    {status !== "empty" && others.length > 0 && (
                      <p className="font-mono text-[11px] text-slate-400 truncate">
                        diğer: {others.map((t) => `${t.name} %${(t.probability * 100).toFixed(0)}`).join(" · ")}
                      </p>
                    )}
                  </div>
                  {status !== "empty" && best && (
                    <span className={`rounded-lg bg-black/75 px-3 py-2 font-mono text-xl font-semibold ${status === "sure" ? "text-cyan-300" : "text-amber-200"}`} data-testid="live-camera-live-confidence">
                      %{(best.probability * 100).toFixed(0)}
                    </span>
                  )}
                </div>
              )}
              {auto && liveError && (
                <div className="absolute left-3 right-3 bottom-3 rounded-lg bg-red-950/85 px-3 py-2 text-sm text-red-200" data-testid="live-camera-live-error">
                  Canlı tanıma hatası: {liveError}
                </div>
              )}
              {auto && ready && !liveError && frames.length === 0 && (
                <div className="absolute left-3 top-3 rounded-lg bg-black/75 px-3 py-2 text-sm text-slate-300 flex items-center gap-2">
                  <Loader2 className="h-4 w-4 animate-spin" /> Model hazırlanıyor…
                </div>
              )}
              {scanning && (
                <div className="pointer-events-none absolute inset-0">
                  <div className="absolute left-0 right-0 h-[3px] bg-cyan-300 shadow-[0_0_18px_4px_rgba(0,240,255,0.7)] animate-scan-beam" />
                </div>
              )}
              {showResult && (
                <div className="absolute left-3 right-3 bottom-3 rounded-lg bg-black/75 backdrop-blur px-4 py-3 flex items-center justify-between gap-3" data-testid="live-camera-result">
                  <div className="min-w-0">
                    <p className="font-mono text-[10px] uppercase tracking-[0.18em] text-cyan-300/80">CNN sonucu</p>
                    <p className="font-heading text-lg font-semibold text-slate-50 truncate" data-testid="live-camera-result-name">{lastResult.name}</p>
                    {lastResult.resistor?.value && (
                      <p className="font-mono text-xs text-amber-300" data-testid="live-camera-result-ohms">{lastResult.resistor.value.formatted} ±%{lastResult.resistor.value.tolerance_percent}</p>
                    )}
                  </div>
                  <span className="font-mono text-2xl font-semibold text-cyan-300" data-testid="live-camera-result-confidence">%{(lastResult.confidence * 100).toFixed(1)}</span>
                </div>
              )}
            </>
          )}
        </div>

        <div className="flex flex-col sm:flex-row sm:flex-wrap gap-3">
          <Button size="lg" className="h-12 flex-1 font-semibold" onClick={capture} disabled={!ready || scanning || !!error} data-testid="live-camera-capture-button">
            {scanning ? <><Loader2 className="h-5 w-5 animate-spin" /> Taranıyor…</> : <><ScanLine className="h-5 w-5" /> Kareyi Tara</>}
          </Button>
          <Button variant="outline" size="lg" className="h-12" onClick={() => setAuto((a) => !a)} disabled={!!error} data-testid="live-camera-auto-toggle">
            {auto ? <><Pause className="h-4 w-4" /> Canlı Tanımayı Durdur</> : <><Play className="h-4 w-4" /> Canlı Tanımayı Başlat</>}
          </Button>
          <Button variant="outline" size="lg" className="h-12" onClick={() => setFacing((f) => (f === "environment" ? "user" : "environment"))} disabled={scanning} data-testid="live-camera-flip-button">
            <RefreshCw className="h-4 w-4" /> Kamerayı Çevir
          </Button>
          <Button variant="outline" size="lg" className="h-12" onClick={() => onOpenChange(false)} disabled={!lastResult || !captured} data-testid="live-camera-details-button">
            <Camera className="h-4 w-4" /> Detayları Gör
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}
