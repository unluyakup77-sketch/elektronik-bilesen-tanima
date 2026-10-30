import type { ReactNode } from "react";
import { NavLink } from "react-router-dom";
import { Cpu } from "lucide-react";
import { cn } from "@/lib/utils";
import { Toaster } from "@/components/ui/sonner";

const LINKS = [
  { to: "/", label: "Bileşen Tara", testId: "nav-scan-link" },
  { to: "/kutuphane", label: "Bileşen Kütüphanesi", testId: "nav-library-link" },
  { to: "/model", label: "Model & Metrikler", testId: "nav-model-link" },
];

export function Layout({ children }: { children: ReactNode }) {
  return (
    <div className="min-h-screen tech-grid flex flex-col">
      <header className="sticky top-0 z-40 border-b border-slate-800/80 bg-[#0B0F19]/80 backdrop-blur-xl">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 h-16 flex items-center gap-6">
          <NavLink to="/" className="flex items-center gap-3 shrink-0" data-testid="brand-home-link">
            <span className="grid place-items-center h-9 w-9 rounded-md border border-cyan-400/40 bg-cyan-400/10 text-cyan-300 shadow-[0_0_18px_rgba(0,240,255,0.25)]">
              <Cpu className="h-5 w-5" />
            </span>
            <span className="leading-tight">
              <span className="block font-heading font-bold tracking-tight text-slate-100">ElektroVision</span>
              <span className="hidden sm:block font-mono text-[10px] uppercase tracking-[0.18em] text-slate-400">EEM · Derin Öğrenme Projesi</span>
            </span>
          </NavLink>
          <nav className="flex items-center gap-1 overflow-x-auto ml-auto">
            {LINKS.map((l) => (
              <NavLink
                key={l.to}
                to={l.to}
                end={l.to === "/"}
                data-testid={l.testId}
                className={({ isActive }) =>
                  cn(
                    "whitespace-nowrap rounded-md px-3 py-2 text-sm font-medium transition-colors duration-150 focus-visible:ring-2 focus-visible:ring-cyan-400 focus-visible:outline-none",
                    isActive ? "bg-cyan-400/10 text-cyan-300" : "text-slate-400 hover:text-slate-100 hover:bg-slate-800/60",
                  )
                }
              >
                {l.label}
              </NavLink>
            ))}
          </nav>
        </div>
      </header>
      <main className="flex-1 mx-auto w-full max-w-7xl px-4 sm:px-6 lg:px-8 py-8 lg:py-12">{children}</main>
      <footer className="border-t border-slate-800/80 py-6">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row gap-2 justify-between text-xs text-slate-500 font-mono">
          <span>ElektroVision · PyTorch ConvNeXt-Tiny CNN · OpenCV</span>
          <span>Elektrik-Elektronik Mühendisliği · Derin Öğrenme</span>
        </div>
      </footer>
      <Toaster richColors />
    </div>
  );
}

export function SectionLabel({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <span className={cn("font-mono text-[11px] font-semibold uppercase tracking-[0.18em] text-cyan-300/80", className)}>
      {children}
    </span>
  );
}
