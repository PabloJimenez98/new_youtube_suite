/**
 * Chart palette and shared ECharts styling (docs/decisions/0022 §5).
 *
 * The app is dark-only, so these are the dataviz reference palette's dark
 * steps, validated against our panel surface #15122f with the skill's
 * validator (all checks pass; first 3 slots also pass all-pairs, which is
 * why scatter charts use at most 3 categorical colours). Canvas can't read
 * CSS variables, so the values live here, next to the design tokens they
 * mirror in styles/_tokens.scss.
 */
export const SERIES = ['#3987e5', '#d95926', '#199e70', '#c98500', '#d55181', '#008300', '#9085e9', '#e66767'];

/** Single-hue sequential ramp (blue), light → dark. */
export const SEQ = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#104281'];

/** Diverging pair with a neutral midpoint: below expectation ← gray → above. */
export const DIVERGING = { low: '#e66767', mid: '#3a3752', high: '#3987e5' };

export const INK = {
  primary: '#f5f3ff',
  secondary: '#b8b3d9',
  muted: '#8b86ad',
  grid: 'rgba(255,255,255,0.07)',
  axis: 'rgba(255,255,255,0.22)',
  surface: '#15122f',
  /** Context marks: everything that isn't the highlighted entity. */
  context: '#5d5880',
};

/** The owner's channel is always this colour; everything else is context gray. */
export const OWN = SERIES[0];

export const FONT = "'Space Grotesk', 'Segoe UI', sans-serif";

export function axisStyle(extra: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    axisLine: { lineStyle: { color: INK.axis } },
    axisTick: { show: false },
    axisLabel: { color: INK.muted, fontFamily: FONT, fontSize: 11 },
    splitLine: { lineStyle: { color: INK.grid } },
    nameTextStyle: { color: INK.muted, fontFamily: FONT, fontSize: 11 },
    ...extra,
  };
}

export function tooltipStyle(extra: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    backgroundColor: '#1f1b45',
    borderColor: 'rgba(255,255,255,0.18)',
    borderWidth: 1,
    padding: [8, 10],
    textStyle: { color: INK.primary, fontFamily: FONT, fontSize: 12 },
    extraCssText: 'box-shadow: 0 8px 30px rgba(0,0,0,0.45); border-radius: 8px;',
    ...extra,
  };
}

export const BASE = {
  backgroundColor: 'transparent',
  textStyle: { fontFamily: FONT, color: INK.secondary },
  animationDuration: 400,
};

/** 12,345 → "12.3K"; 1,234,567 → "1.23M". */
export function compact(n: number | null | undefined): string {
  if (n == null || Number.isNaN(n)) return '–';
  const abs = Math.abs(n);
  if (abs >= 1e9) return `${+(n / 1e9).toPrecision(3)}B`;
  if (abs >= 1e6) return `${+(n / 1e6).toPrecision(3)}M`;
  if (abs >= 1e3) return `${+(n / 1e3).toPrecision(3)}K`;
  return `${+n.toPrecision(3)}`;
}

export function pct(n: number | null | undefined, digits = 1): string {
  return n == null ? '–' : `${(n * 100).toFixed(digits)}%`;
}

export function escapeHtml(s: string): string {
  return s.replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]!);
}
