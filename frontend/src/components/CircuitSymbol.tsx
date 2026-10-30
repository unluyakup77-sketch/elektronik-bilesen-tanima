import type { ReactNode } from "react";

// IEC/ANSI karışık, ders kitaplarında yaygın devre sembolleri. viewBox 0 0 120 60.
const ZIGZAG = "M0 30 H30 L34 18 L42 42 L50 18 L58 42 L66 18 L74 42 L82 18 L86 30 H120";

function mcuBlock(label: string) {
  return (
    <>
      <rect x="30" y="6" width="60" height="48" rx="3" />
      <path d="M16 14 H30 M16 24 H30 M16 36 H30 M16 46 H30 M90 14 H104 M90 24 H104 M90 36 H104 M90 46 H104" />
      <text x="60" y="35" textAnchor="middle" fontSize={label.length > 4 ? 11 : 13} fill="currentColor" stroke="none" fontFamily="JetBrains Mono Variable, monospace">{label}</text>
    </>
  );
}

const SYMBOLS: Record<string, ReactNode> = {
  arduino: mcuBlock("MCU"),
  esp32: (
    <>
      {mcuBlock("ESP32")}
      <path d="M100 2 V-2 M96 -2 L100 2 L104 -2" />
    </>
  ),
  stm32: mcuBlock("STM32"),
  direnc: <path d={ZIGZAG} />,
  led: (
    <>
      <path d="M0 30 H45 M75 30 H120 M45 16 V44 L73 30 Z M75 16 V44" />
      <path d="M62 12 L74 2 M70 2 H74 V6 M72 18 L84 8 M80 8 H84 V12" />
    </>
  ),
  seramik_kondansator: <path d="M0 30 H54 M66 30 H120 M54 12 V48 M66 12 V48" />,
  elektrolitik_kondansator: (
    <>
      <path d="M0 30 H54 M64 30 H120 M54 12 V48" />
      <path d="M70 12 Q62 30 70 48" />
      <path d="M40 12 H48 M44 8 V16" />
    </>
  ),
  transistor: (
    <>
      <circle cx="64" cy="30" r="24" />
      <path d="M0 30 H54 M54 16 V44 M54 22 L76 8 V0 M54 38 L76 52 V60" />
      <path d="M76 52 L66 50 M76 52 L71 44" />
    </>
  ),
  diyot: <path d="M0 30 H45 M75 30 H120 M45 14 V46 L75 30 Z M75 14 V46" />,
  entegre: (
    <>
      <rect x="35" y="8" width="50" height="44" rx="2" />
      <path d="M55 8 A5 5 0 0 0 65 8" />
      <path d="M20 18 H35 M20 30 H35 M20 42 H35 M85 18 H100 M85 30 H100 M85 42 H100" />
      <text x="60" y="35" textAnchor="middle" fontSize="12" fill="currentColor" stroke="none" fontFamily="JetBrains Mono Variable, monospace">IC</text>
    </>
  ),
  potansiyometre: (
    <>
      <path d="M0 30 H30 L34 20 L42 40 L50 20 L58 40 L66 20 L74 40 L82 20 L86 30 H120" />
      <path d="M58 58 V46 M52 50 L58 42 L64 50" />
    </>
  ),
  bobin: <path d="M0 34 H24 A9 9 0 0 1 42 34 A9 9 0 0 1 60 34 A9 9 0 0 1 78 34 A9 9 0 0 1 96 34 H120" />,
  role: (
    <>
      <rect x="6" y="18" width="34" height="24" />
      <path d="M6 30 L40 30" strokeDasharray="0" opacity="0.5" />
      <path d="M23 0 V18 M23 42 V60" />
      <path d="M40 30 H52" strokeDasharray="3 3" />
      <path d="M60 50 V40 L88 22 M88 40 V50 M60 10 V40 M88 40 H96 V10" />
      <circle cx="88" cy="40" r="2.5" />
    </>
  ),
  sigorta: (
    <>
      <rect x="32" y="20" width="56" height="20" rx="2" />
      <path d="M0 30 H120" />
    </>
  ),
  transformator: (
    <>
      <path d="M40 4 A8 8 0 0 1 40 20 A8 8 0 0 1 40 36 A8 8 0 0 1 40 52" />
      <path d="M80 4 A8 8 0 0 0 80 20 A8 8 0 0 0 80 36 A8 8 0 0 0 80 52" />
      <path d="M56 2 V58 M64 2 V58" />
      <path d="M0 4 H40 M0 52 H40 M80 4 H120 M80 52 H120" />
    </>
  ),
  buzzer: (
    <>
      <path d="M40 40 A20 20 0 0 1 80 40 Z" />
      <path d="M0 40 H40 M80 40 H120" />
      <path d="M56 40 V54 M64 40 V54" />
    </>
  ),
  kristal_osilator: (
    <>
      <path d="M0 30 H44 M76 30 H120 M44 14 V46 M76 14 V46" />
      <rect x="52" y="16" width="16" height="28" />
    </>
  ),
  buton: (
    <>
      <path d="M0 44 H36 M84 44 H120" />
      <circle cx="38" cy="44" r="2.5" />
      <circle cx="82" cy="44" r="2.5" />
      <path d="M32 32 H88 M60 32 V10 M50 10 H70" />
    </>
  ),
  display_7seg: (
    <>
      <rect x="36" y="2" width="48" height="56" rx="3" opacity="0.5" />
      <path d="M48 8 H72 M46 10 V28 M74 10 V28 M48 30 H72 M46 32 V50 M74 32 V50 M48 52 H72" />
      <circle cx="80" cy="53" r="1.5" />
    </>
  ),
};

interface Props {
  type: string;
  className?: string;
  testId?: string;
}

export function CircuitSymbol({ type, className, testId }: Props) {
  const sym = SYMBOLS[type];
  return (
    <svg
      viewBox="-4 -4 128 68"
      fill="none"
      stroke="currentColor"
      strokeWidth={2}
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
      role="img"
      aria-label={`${type} devre sembolü`}
      data-testid={testId}
    >
      {sym ?? (
        <text x="60" y="36" textAnchor="middle" fontSize="14" fill="currentColor" stroke="none">
          ?
        </text>
      )}
    </svg>
  );
}
