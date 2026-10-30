import { motion } from "motion/react";
import { AlertTriangle, BrainCircuit, CheckCircle2, Cpu, Lightbulb, ListChecks, Package, Sparkles, Zap } from "lucide-react";
import { CircuitSymbol } from "@/components/CircuitSymbol";
import { FeedbackCard } from "@/components/FeedbackCard";
import { ResistorDecoder } from "@/components/ResistorDecoder";
import { SectionLabel } from "@/components/Layout";
import type { Prediction } from "@/lib/types";
import { cn } from "@/lib/utils";

const pct = (v: number) => `%${(v * 100).toFixed(1)}`;

function ConfidenceRing({ value }: { value: number }) {
  const r = 34;
  const c = 2 * Math.PI * r;
  const color = value >= 0.85 ? "#10B981" : value >= 0.6 ? "#F59E0B" : "#EF4444";
  return (
    <div className="relative h-24 w-24 shrink-0" data-testid="result-confidence-ring">
      <svg viewBox="0 0 80 80" className="h-full w-full -rotate-90">
        <circle cx="40" cy="40" r={r} stroke="#1F2937" strokeWidth="7" fill="none" />
        <motion.circle
          cx="40" cy="40" r={r} stroke={color} strokeWidth="7" fill="none" strokeLinecap="round"
          strokeDasharray={c}
          initial={{ strokeDashoffset: c }}
          animate={{ strokeDashoffset: c * (1 - value) }}
          transition={{ duration: 0.8, ease: "easeOut" }}
        />
      </svg>
      <span className="absolute inset-0 grid place-items-center font-mono text-lg font-semibold text-slate-100" data-testid="result-confidence">
        {pct(value)}
      </span>
    </div>
  );
}

export function ScanResult({ result }: { result: Prediction }) {
  const info = result.info;
  return (
    <motion.div
      key={result.id}
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35, ease: "easeOut" }}
      className="flex flex-col gap-5"
      data-testid="scan-result"
    >
      {/* Başlık kartı */}
      <div className="rounded-xl border border-slate-800 bg-[#111827]/90 p-5 sm:p-6 shadow-[0_0_0_1px_rgba(0,240,255,0.06)]">
        <div className="flex items-start gap-5">
          <ConfidenceRing value={result.confidence} />
          <div className="min-w-0 flex-1">
            <SectionLabel>Tespit edilen bileşen</SectionLabel>
            <h2 className="mt-1 text-3xl sm:text-4xl font-bold tracking-tight text-slate-50" data-testid="result-component-name">
              {result.name}
            </h2>
            {info && <p className="mt-1 font-mono text-xs text-slate-400">{info.english_name} · {info.category}</p>}
            <div className="mt-3 flex flex-wrap gap-2">
              {result.source === "cnn" ? (
                <span data-testid="result-source-badge" className="inline-flex items-center gap-1.5 rounded-md bg-[#0E3A52] px-2.5 py-1 font-mono text-[11px] font-semibold uppercase tracking-wider text-sky-300">
                  <BrainCircuit className="h-3.5 w-3.5" /> CNN kararı
                </span>
              ) : (
                <span data-testid="result-source-badge" className="inline-flex items-center gap-1.5 rounded-md bg-[#2D1B4E] px-2.5 py-1 font-mono text-[11px] font-semibold uppercase tracking-wider text-purple-300">
                  <Sparkles className="h-3.5 w-3.5" /> CNN + AI doğrulama
                </span>
              )}
              {result.agreement === true && (
                <span className="inline-flex items-center gap-1.5 rounded-md bg-emerald-500/10 px-2.5 py-1 font-mono text-[11px] font-semibold uppercase tracking-wider text-emerald-300" data-testid="result-agreement-badge">
                  <CheckCircle2 className="h-3.5 w-3.5" /> İki model hemfikir
                </span>
              )}
              {result.agreement === false && (
                <span className="inline-flex items-center gap-1.5 rounded-md bg-amber-500/10 px-2.5 py-1 font-mono text-[11px] font-semibold uppercase tracking-wider text-amber-300" data-testid="result-agreement-badge">
                  <AlertTriangle className="h-3.5 w-3.5" /> AI, CNN'i düzeltti
                </span>
              )}
            </div>
          </div>
        </div>
        {result.ai_note && (
          <div className="mt-5 rounded-lg border border-purple-500/20 bg-purple-500/5 p-3 text-sm text-slate-300 leading-relaxed" data-testid="result-ai-note">
            <span className="font-mono text-[11px] uppercase tracking-wider text-purple-300 mr-2">AI notu</span>
            {result.ai_note}
          </div>
        )}
        {result.source === "cnn" && result.cnn_confidence < 0.6 && (
          <div className="mt-5 rounded-lg border border-amber-500/30 bg-amber-500/5 p-3 text-sm text-slate-300" data-testid="result-low-confidence-hint">
            <span className="font-semibold text-amber-300">CNN bu fotoğrafta emin değil. </span>
            Bileşeni yakından, düz ve sade bir zeminde, iyi ışıkta tekrar çekmeyi dene. Aşağıdaki diğer olası tahminlere de göz at.
          </div>
        )}
        {!result.is_component && (
          <div className="mt-5 rounded-lg border border-red-500/30 bg-red-500/5 p-4 text-sm text-slate-300" data-testid="result-unknown-warning">
            <div className="flex items-center gap-2 font-semibold text-red-300"><AlertTriangle className="h-4 w-4" /> Bileşen tanımlanamadı</div>
            <p className="mt-1">Fotoğrafta tanıdığımız 19 bileşenden biri net olarak görülmüyor. Bileşeni düz bir zemine koyup yakından, iyi ışıkta tekrar çekmeyi deneyin.</p>
          </div>
        )}
      </div>

      {/* CNN olasılıkları */}
      <div className="rounded-xl border border-slate-800 bg-[#111827]/90 p-5 sm:p-6">
        <div className="flex items-center justify-between">
          <SectionLabel>CNN · İlk 3 tahmin (softmax)</SectionLabel>
          {result.ai_used && result.ai_confidence !== null && (
            <span className="font-mono text-[11px] text-purple-300" data-testid="result-ai-confidence">AI güveni: {pct(result.ai_confidence)}</span>
          )}
        </div>
        <ul className="mt-4 flex flex-col gap-3" data-testid="result-top3-list">
          {result.cnn_top3.map((p, i) => (
            <li key={p.key} data-testid={`result-top3-item-${i}`}>
              <div className="flex justify-between text-sm">
                <span className={cn("font-medium", i === 0 ? "text-slate-100" : "text-slate-400")}>{i + 1}. {p.name}</span>
                <span className="font-mono text-slate-300">{pct(p.probability)}</span>
              </div>
              <div className="mt-1.5 h-2 rounded-full bg-slate-800 overflow-hidden" role="progressbar" aria-valuenow={Math.round(p.probability * 100)} aria-valuemin={0} aria-valuemax={100}>
                <motion.div
                  className={cn("h-full rounded-full", i === 0 ? "bg-cyan-400 shadow-[0_0_12px_rgba(0,240,255,0.6)]" : "bg-slate-500")}
                  initial={{ width: 0 }}
                  animate={{ width: `${Math.max(1, p.probability * 100)}%` }}
                  transition={{ duration: 0.6, ease: "easeOut", delay: i * 0.08 }}
                />
              </div>
            </li>
          ))}
        </ul>
      </div>

      <FeedbackCard result={result} />

      {result.resistor && <ResistorDecoder reading={result.resistor} />}

      {info?.how_it_works && (
        <div className="rounded-xl border border-cyan-500/20 bg-[#111827]/90 p-5 sm:p-6" data-testid="result-how-it-works">
          <SectionLabel className="flex items-center gap-2"><Cpu className="h-3.5 w-3.5" /> Nasıl çalışır?</SectionLabel>
          <p className="mt-3 text-slate-300 leading-relaxed">{info.how_it_works}</p>
          {info.specs.length > 0 && (
            <ul className="mt-4 grid grid-cols-1 sm:grid-cols-2 gap-2" data-testid="result-specs-list">
              {info.specs.map((sp) => (
                <li key={sp} className="rounded-md border border-slate-800 bg-[#0d1424] px-3 py-2 font-mono text-xs text-slate-300">{sp}</li>
              ))}
            </ul>
          )}
        </div>
      )}

      {info && (
        <div className="grid grid-cols-1 md:grid-cols-5 gap-5">
          <div className="md:col-span-3 rounded-xl border border-slate-800 bg-[#111827]/90 p-5 sm:p-6">
            <SectionLabel className="flex items-center gap-2"><Zap className="h-3.5 w-3.5" /> Ne işe yarar?</SectionLabel>
            <p className="mt-3 text-slate-300 leading-relaxed" data-testid="result-function-text">{info.function}</p>
            <p className="mt-3 font-mono text-xs text-slate-400" data-testid="result-typical-values">Tipik değerler: {info.typical_values}</p>
          </div>
          <div className="md:col-span-2 rounded-xl border border-slate-800 bg-[#0d1424] p-5 sm:p-6 flex flex-col">
            <SectionLabel>Devre sembolü</SectionLabel>
            <div className="flex-1 grid place-items-center py-4">
              <CircuitSymbol type={info.key} testId="result-circuit-symbol" className="w-full max-w-[220px] text-cyan-300 drop-shadow-[0_0_8px_rgba(0,240,255,0.45)]" />
            </div>
            <div className="flex flex-wrap gap-1.5">
              {info.packages.map((p) => (
                <span key={p} className="inline-flex items-center gap-1 rounded border border-slate-700 px-2 py-0.5 font-mono text-[10px] text-slate-400"><Package className="h-3 w-3" />{p}</span>
              ))}
            </div>
          </div>
          <div className="md:col-span-3 rounded-xl border border-slate-800 bg-[#111827]/90 p-5 sm:p-6">
            <SectionLabel className="flex items-center gap-2"><ListChecks className="h-3.5 w-3.5" /> Kullanım alanları</SectionLabel>
            <ul className="mt-3 grid grid-cols-1 sm:grid-cols-2 gap-2" data-testid="result-usage-list">
              {info.usage_areas.map((u) => (
                <li key={u} className="flex gap-2 text-sm text-slate-300"><span className="mt-2 h-1.5 w-1.5 rounded-full bg-cyan-400 shrink-0" />{u}</li>
              ))}
            </ul>
          </div>
          <div className="md:col-span-2 rounded-xl border border-slate-800 bg-[#111827]/90 p-5 sm:p-6">
            <SectionLabel className="flex items-center gap-2"><Lightbulb className="h-3.5 w-3.5" /> Nasıl ayırt edilir?</SectionLabel>
            <ul className="mt-3 flex flex-col gap-2" data-testid="result-tips-list">
              {info.identify_tips.map((t) => (
                <li key={t} className="text-sm text-slate-400 leading-relaxed">— {t}</li>
              ))}
            </ul>
          </div>
        </div>
      )}
    </motion.div>
  );
}
