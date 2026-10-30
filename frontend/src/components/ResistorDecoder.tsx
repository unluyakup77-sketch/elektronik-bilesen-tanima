import { useState } from "react";
import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { Calculator, Info } from "lucide-react";
import { SectionLabel } from "@/components/Layout";
import { apiGet } from "@/lib/api";
import type { ResistorReading, ResistorValue } from "@/lib/types";
import { cn } from "@/lib/utils";

const COLORS: Record<string, { tr: string; hex: string }> = {
  black: { tr: "Siyah", hex: "#111111" },
  brown: { tr: "Kahverengi", hex: "#8B4513" },
  red: { tr: "Kırmızı", hex: "#E11D48" },
  orange: { tr: "Turuncu", hex: "#F97316" },
  yellow: { tr: "Sarı", hex: "#FACC15" },
  green: { tr: "Yeşil", hex: "#16A34A" },
  blue: { tr: "Mavi", hex: "#2563EB" },
  violet: { tr: "Mor", hex: "#8B5CF6" },
  grey: { tr: "Gri", hex: "#9CA3AF" },
  white: { tr: "Beyaz", hex: "#F8FAFC" },
  gold: { tr: "Altın", hex: "#D4AF37" },
  silver: { tr: "Gümüş", hex: "#C0C0C0" },
};
const DIGIT = ["black", "brown", "red", "orange", "yellow", "green", "blue", "violet", "grey", "white"];
const MULT = [...DIGIT, "gold", "silver"];
const TOL = ["brown", "red", "green", "blue", "violet", "grey", "gold", "silver"];

function roleOf(i: number, n: number) {
  if (i === n - 1) return { label: "Tolerans", options: TOL };
  if (i === n - 2) return { label: "Çarpan", options: MULT };
  return { label: `${i + 1}. rakam`, options: DIGIT };
}

export function ResistorDecoder({ reading }: { reading: ResistorReading }) {
  const [bands, setBands] = useState<string[]>(reading.bands);
  const [active, setActive] = useState<number | null>(null);
  // Otomatik okuma başarısızsa, kullanıcı bant seçene kadar değer gösterilmez (yanlış değer göstermemek için)
  const [touched, setTouched] = useState(reading.detected);
  const key = bands.join(",");
  const original = reading.bands.join(",");

  const value = useQuery({
    queryKey: ["resistor-value", key],
    queryFn: () => apiGet<ResistorValue>(`/resistor/value?bands=${key}`),
    enabled: touched,
    initialData: key === original && reading.value ? reading.value : undefined,
    placeholderData: keepPreviousData,
  });

  function setCount(n: 4 | 5) {
    if (n === bands.length) return;
    setActive(null);
    setTouched(true);
    setBands(n === 5 ? [bands[0], bands[1], "black", bands[2], bands[3]] : [bands[0], bands[1], bands[3], bands[4]]);
  }

  function choose(color: string) {
    if (active === null) return;
    setTouched(true);
    setBands(bands.map((b, i) => (i === active ? color : b)));
  }

  const v = touched && !value.isError ? value.data : undefined;
  const activeRole = active !== null ? roleOf(active, bands.length) : null;

  return (
    <div className="rounded-xl border border-amber-500/20 bg-[#111827]/90 p-5 sm:p-6" data-testid="resistor-decoder">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <SectionLabel className="flex items-center gap-2 text-amber-300/90"><Calculator className="h-3.5 w-3.5" /> Renk kodu → Ohm değeri</SectionLabel>
        <div className="flex rounded-md border border-slate-700 overflow-hidden font-mono text-xs" role="group" aria-label="Bant sayısı">
          {([4, 5] as const).map((n) => (
            <button
              key={n}
              type="button"
              onClick={() => setCount(n)}
              data-testid={`resistor-band-count-${n}`}
              className={cn("px-3 py-1.5 transition-colors", bands.length === n ? "bg-amber-400/15 text-amber-300" : "text-slate-400 hover:text-slate-100")}
            >
              {n} bant
            </button>
          ))}
        </div>
      </div>

      <div className="mt-5 grid grid-cols-1 sm:grid-cols-2 gap-6 items-center">
        {/* Direnç çizimi */}
        <div className="relative h-20 flex items-center" aria-label="Direnç bantları">
          <div className="absolute left-0 right-0 h-1 bg-slate-500 rounded" />
          <div className="relative mx-6 flex-1 h-14 rounded-full bg-[#E8D3A9] flex items-center justify-around px-5 shadow-inner">
            {bands.map((b, i) => (
              <button
                key={i}
                type="button"
                onClick={() => setActive(active === i ? null : i)}
                data-testid={`resistor-band-${i}`}
                aria-label={`${roleOf(i, bands.length).label}: ${COLORS[b]?.tr ?? b}`}
                className={cn(
                  "h-full w-3.5 sm:w-4 rounded-sm transition-transform duration-150 hover:scale-x-125 focus-visible:outline-none",
                  i === bands.length - 1 && "ml-3",
                  active === i && "ring-2 ring-cyan-300 ring-offset-2 ring-offset-[#E8D3A9] scale-x-125",
                )}
                style={{ background: COLORS[b]?.hex ?? "#555" }}
              />
            ))}
          </div>
        </div>

        {/* Değer */}
        <div>
          <p className="font-mono text-4xl font-semibold text-slate-50" data-testid="resistor-value">
            {v ? v.formatted : "—"}
          </p>
          {!touched && <p className="mt-1 text-sm text-amber-300" data-testid="resistor-manual-hint">Bantlara tıklayıp renkleri seç, değer hemen hesaplansın.</p>}
          {v && (
            <p className="mt-1 font-mono text-sm text-slate-400" data-testid="resistor-tolerance">
              ±%{v.tolerance_percent} · {v.formula}
            </p>
          )}
          {v && <p className="mt-0.5 font-mono text-xs text-slate-500">Aralık: {fmt(v.min_ohms)} – {fmt(v.max_ohms)}</p>}
        </div>
      </div>

      <p className="mt-3 text-xs text-slate-500">
        Okunan bantlar: <span className="text-slate-300" data-testid="resistor-band-names">{bands.map((b) => COLORS[b]?.tr ?? b).join(" · ")}</span>
      </p>

      {activeRole && (
        <div className="mt-4 rounded-lg border border-slate-800 bg-[#0d1424] p-3" data-testid="resistor-color-palette">
          <p className="text-xs text-slate-400 mb-2">{activeRole.label} için renk seç:</p>
          <div className="flex flex-wrap gap-2">
            {activeRole.options.map((c) => (
              <button
                key={c}
                type="button"
                onClick={() => choose(c)}
                data-testid={`resistor-color-${c}`}
                className={cn(
                  "flex items-center gap-1.5 rounded-md border px-2 py-1 text-xs transition-colors",
                  active !== null && bands[active] === c ? "border-cyan-400 text-cyan-200" : "border-slate-700 text-slate-300 hover:border-slate-500",
                )}
              >
                <span className="h-3.5 w-3.5 rounded-sm border border-black/30" style={{ background: COLORS[c].hex }} />
                {COLORS[c].tr}
              </button>
            ))}
          </div>
        </div>
      )}

      <div className="mt-4 flex gap-2 text-xs text-slate-400 leading-relaxed" data-testid="resistor-note">
        <Info className="h-4 w-4 shrink-0 text-amber-300/80" />
        <span>
          {reading.note ? `${reading.note} ` : ""}Ohm değeri Python ile hesaplanır{reading.detected ? "; yanlışsa bir banda tıklayıp rengini değiştir." : "."}
        </span>
      </div>
    </div>
  );
}

function fmt(ohms: number) {
  if (ohms >= 1e6) return `${+(ohms / 1e6).toFixed(3)} MΩ`;
  if (ohms >= 1e3) return `${+(ohms / 1e3).toFixed(3)} kΩ`;
  return `${+ohms.toFixed(2)} Ω`;
}
