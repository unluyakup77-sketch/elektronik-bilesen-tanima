import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { motion } from "motion/react";
import { Search } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Layout, SectionLabel } from "@/components/Layout";
import { CircuitSymbol } from "@/components/CircuitSymbol";
import { apiGet } from "@/lib/api";
import type { ComponentInfo } from "@/lib/types";
import { cn } from "@/lib/utils";

const CATEGORIES = ["Tümü", "Pasif", "Aktif", "Elektromekanik", "Opto / Gösterge", "Koruma", "Mikrodenetleyici"];

const slug = (s: string) => s.toLowerCase().replace(/[^a-z0-9]+/g, "-");

export default function Library() {
  const [q, setQ] = useState("");
  const [cat, setCat] = useState("Tümü");
  const [open, setOpen] = useState<ComponentInfo | null>(null);
  const { data, isError, isLoading } = useQuery({ queryKey: ["components"], queryFn: () => apiGet<ComponentInfo[]>("/components") });

  const items = useMemo(() => {
    const list = isError ? [] : (data ?? []);
    const term = q.trim().toLocaleLowerCase("tr");
    return list.filter(
      (c) =>
        (cat === "Tümü" || c.category === cat) &&
        (!term || `${c.name} ${c.english_name} ${c.summary}`.toLocaleLowerCase("tr").includes(term)),
    );
  }, [data, isError, q, cat]);

  return (
    <Layout>
      <section className="max-w-3xl">
        <SectionLabel>Referans katalog</SectionLabel>
        <h1 className="mt-3 text-4xl sm:text-5xl font-bold tracking-tight text-slate-50">Bileşen Kütüphanesi</h1>
        <p className="mt-3 text-slate-400 leading-relaxed">Modelimizin tanıyabildiği 19 bileşen sınıfı. Her birinin devre sembolünü, ne işe yaradığını ve nasıl ayırt edileceğini incele.</p>
      </section>

      <div className="mt-8 flex flex-col md:flex-row gap-4 md:items-center">
        <div className="relative md:w-80">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-500" />
          <Input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Bileşen ara (ör. kondansatör)" className="pl-9 h-11 bg-[#111827]" data-testid="library-search-input" />
        </div>
        <div className="flex gap-2 overflow-x-auto pb-1">
          {CATEGORIES.map((c) => (
            <button
              key={c}
              type="button"
              onClick={() => setCat(c)}
              data-testid={`library-filter-${slug(c)}`}
              className={cn(
                "whitespace-nowrap rounded-md border px-3 py-2 text-sm transition-colors duration-150",
                cat === c ? "border-cyan-400/50 bg-cyan-400/10 text-cyan-300" : "border-slate-800 text-slate-400 hover:text-slate-100 hover:border-slate-600",
              )}
            >
              {c}
            </button>
          ))}
        </div>
      </div>

      <div className="mt-8 grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-5" data-testid="library-grid">
        {isLoading && Array.from({ length: 8 }).map((_, i) => <div key={i} className="h-60 rounded-xl border border-slate-800 bg-[#111827] animate-pulse" />)}
        {isError && <p className="text-sm text-slate-500 col-span-full">Kütüphane şu an yüklenemedi.</p>}
        {items.map((c, i) => (
          <motion.button
            key={c.key}
            type="button"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.25, delay: Math.min(i * 0.03, 0.3) }}
            onClick={() => setOpen(c)}
            data-testid={`library-card-${c.key}`}
            className="group flex flex-col justify-start text-left rounded-xl border border-slate-800 bg-[#111827]/90 p-5 hover:border-cyan-500/40 hover:-translate-y-0.5 transition-[border-color,transform] duration-200"
          >
            <div className="h-24 rounded-lg bg-[#0d1424] border border-slate-800/80 grid place-items-center">
              <CircuitSymbol type={c.key} className="h-16 w-36 text-cyan-300/90 group-hover:drop-shadow-[0_0_6px_rgba(0,240,255,0.5)] transition-[filter]" />
            </div>
            <span className="mt-4 block font-mono text-[10px] uppercase tracking-[0.18em] text-slate-500">{c.category}</span>
            <span className="mt-1 block font-heading text-lg font-semibold text-slate-100">{c.name}</span>
            <span className="block font-mono text-xs text-slate-500">{c.english_name}</span>
            <span className="mt-2 block text-sm text-slate-400 leading-relaxed line-clamp-3">{c.summary}</span>
          </motion.button>
        ))}
        {!isLoading && !isError && items.length === 0 && <p className="text-sm text-slate-500 col-span-full" data-testid="library-empty">Aramanla eşleşen bileşen yok.</p>}
      </div>

      <Dialog open={open !== null} onOpenChange={(o) => !o && setOpen(null)}>
        <DialogContent className="sm:max-w-2xl max-h-[90vh] overflow-y-auto bg-[#111827] border-slate-800" data-testid="library-detail-dialog">
          {open && (
            <>
              <DialogHeader>
                <span className="font-mono text-[10px] uppercase tracking-[0.18em] text-cyan-300/80">{open.category} · {open.english_name}</span>
                <DialogTitle className="text-2xl font-heading">{open.name}</DialogTitle>
                <DialogDescription className="text-slate-400">{open.summary}</DialogDescription>
              </DialogHeader>
              <div className="grid place-items-center rounded-lg bg-[#0d1424] border border-slate-800 py-6">
                <CircuitSymbol type={open.key} className="h-20 w-48 text-cyan-300 drop-shadow-[0_0_8px_rgba(0,240,255,0.45)]" />
              </div>
              <div className="space-y-4 text-sm">
                <div>
                  <SectionLabel>Ne işe yarar?</SectionLabel>
                  <p className="mt-1.5 text-slate-300 leading-relaxed">{open.function}</p>
                </div>
                {open.how_it_works && (
                  <div data-testid="library-detail-how-it-works">
                    <SectionLabel>Nasıl çalışır?</SectionLabel>
                    <p className="mt-1.5 text-slate-300 leading-relaxed">{open.how_it_works}</p>
                    {open.specs.length > 0 && (
                      <ul className="mt-2 grid sm:grid-cols-2 gap-1.5">
                        {open.specs.map((sp) => <li key={sp} className="rounded border border-slate-800 bg-[#0d1424] px-2 py-1 font-mono text-[11px] text-slate-300">{sp}</li>)}
                      </ul>
                    )}
                  </div>
                )}
                <div className="grid sm:grid-cols-2 gap-4">
                  <div>
                    <SectionLabel>Kullanım alanları</SectionLabel>
                    <ul className="mt-1.5 space-y-1 text-slate-300">{open.usage_areas.map((u) => <li key={u}>• {u}</li>)}</ul>
                  </div>
                  <div>
                    <SectionLabel>Nasıl ayırt edilir?</SectionLabel>
                    <ul className="mt-1.5 space-y-1 text-slate-400">{open.identify_tips.map((t) => <li key={t}>— {t}</li>)}</ul>
                  </div>
                </div>
                <p className="font-mono text-xs text-slate-400">Tipik değerler: {open.typical_values}</p>
                <p className="font-mono text-xs text-slate-500">Kılıflar: {open.packages.join(" · ")}</p>
              </div>
            </>
          )}
        </DialogContent>
      </Dialog>
    </Layout>
  );
}
