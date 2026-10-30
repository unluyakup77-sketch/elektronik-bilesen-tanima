import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { Check, GraduationCap, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { SectionLabel } from "@/components/Layout";
import { apiGet, apiPost, errorMessage } from "@/lib/api";
import type { ComponentInfo, FeedbackResult, Prediction } from "@/lib/types";

// "Sonuç doğru mu?" — kullanıcının onayı/düzeltmesi fotoğrafı ml/dataset_user/<etiket>/ altına kaydeder.
// Bu gerçek fotoğraflar modeli_guncelle.py ile yapılan ince ayarda eğitime girer (insan-döngüde öğrenme).
export function FeedbackCard({ result }: { result: Prediction }) {
  const qc = useQueryClient();
  const components = useQuery({ queryKey: ["components"], queryFn: () => apiGet<ComponentInfo[]>("/components") });
  const [saved, setSaved] = useState<string | null>(result.feedback_label ?? null);
  const [choice, setChoice] = useState("");

  const send = useMutation({
    mutationFn: (label: string) => apiPost<FeedbackResult>(`/predictions/${result.id}/feedback`, { label }),
    onSuccess: (r, label) => {
      setSaved(label);
      qc.invalidateQueries({ queryKey: ["predictions"] });
      toast.success(`Kaydedildi — toplam ${r.total} öğretici fotoğraf. Model güncellenince bunu da öğrenecek.`);
    },
    onError: (e) => toast.error(errorMessage(e, "Kaydedilemedi.")),
  });

  const options = [...(components.data ?? []).map((c) => ({ key: c.key, name: c.name })), { key: "bilinmiyor", name: "Bileşen değil" }];
  const nameOf = (k: string) => options.find((o) => o.key === k)?.name ?? k;

  return (
    <div className="rounded-xl border border-slate-800 bg-[#111827]/90 p-5 sm:p-6" data-testid="result-feedback">
      <SectionLabel className="flex items-center gap-2"><GraduationCap className="h-3.5 w-3.5" /> Sonuç doğru mu? Modeli birlikte eğitelim</SectionLabel>
      {saved ? (
        <p className="mt-3 text-sm text-emerald-300 flex items-center gap-2" data-testid="feedback-saved">
          <Check className="h-4 w-4" /> Bu fotoğraf "{nameOf(saved)}" olarak eğitim setine eklendi.
        </p>
      ) : (
        <>
          <p className="mt-2 text-sm text-slate-400">
            Onayladığın veya düzelttiğin fotoğraflar modelin bir sonraki güncellemesinde eğitime girer; gerçek kamera fotoğrafları modeli en çok geliştiren veridir.
          </p>
          <div className="mt-4 flex flex-col sm:flex-row gap-3">
            {result.is_component && (
              <Button variant="outline" className="h-11" disabled={send.isPending} onClick={() => send.mutate(result.label)} data-testid="feedback-correct">
                <Check className="h-4 w-4" /> Doğru, bu bir {result.name}
              </Button>
            )}
            <div className="flex flex-1 gap-2">
              <select
                className="h-11 flex-1 rounded-md border border-slate-700 bg-[#0d1424] px-3 text-sm text-slate-200"
                value={choice}
                onChange={(e) => setChoice(e.target.value)}
                data-testid="feedback-select"
              >
                <option value="">Yanlış — doğrusu…</option>
                {options.filter((o) => o.key !== result.label).map((o) => (
                  <option key={o.key} value={o.key}>{o.name}</option>
                ))}
              </select>
              <Button className="h-11" disabled={!choice || send.isPending} onClick={() => send.mutate(choice)} data-testid="feedback-submit">
                {send.isPending ? <Loader2 className="h-4 w-4 animate-spin" /> : "Kaydet"}
              </Button>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
