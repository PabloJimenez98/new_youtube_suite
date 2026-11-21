/**
 * Charts of the own-LLM section of How it works (M16, decision 0025).
 * Pure builders over the generated, measured OWN_LLM_DATA: each returns an
 * ECharts option plus the same values as a table (0022 §5: no value is
 * readable only through a hover tooltip). One y-axis per chart: measures of
 * different units get their own chart.
 */
import type { EChartsCoreOption } from 'echarts/core';

import { ChartTable } from '../charts/chart-card';
import { BASE, DIVERGING, INK, SERIES, axisStyle, pct, tooltipStyle } from '../charts/viz';
import { OWN_LLM_DATA } from './own-llm-data';

export interface Built {
  option: EChartsCoreOption;
  table: ChartTable;
}

const D = OWN_LLM_DATA;
const GRID = { left: 8, right: 20, top: 34, bottom: 28, containLabel: true };
const LEGEND = { top: 0, left: 0, itemWidth: 14, itemHeight: 3, textStyle: { color: INK.secondary, fontSize: 11 } };
const m = (n: number) => `${(n / 1e6).toFixed(1)}M`;

/** Where the 100.7M parameters are: one segmented bar, direct-labelled. */
export function paramBreakdown(): Built {
  const p = D.arch.params;
  const parts = [
    { name: 'Embedding (shared with the output)', value: p.embedding, color: SERIES[0] },
    { name: 'Attention (12 blocks)', value: p.attention, color: SERIES[1] },
    { name: 'Feed-forward MLP (12 blocks)', value: p.ffn, color: SERIES[2] },
  ];
  return {
    option: {
      ...BASE,
      grid: { left: 0, right: 0, top: 30, bottom: 0, containLabel: false },
      legend: { ...LEGEND, data: parts.map((x) => x.name) },
      tooltip: tooltipStyle({ trigger: 'item', formatter: (q: { seriesName: string; value: number }) => `${q.seriesName}<br>${m(q.value)} · ${pct(q.value / p.total, 0)}` }),
      xAxis: { type: 'value', max: p.total, show: false },
      yAxis: { type: 'category', data: ['params'], show: false },
      series: parts.map((x, i) => ({
        name: x.name,
        type: 'bar',
        stack: 'p',
        barWidth: 28,
        data: [x.value],
        itemStyle: {
          color: x.color,
          borderColor: INK.surface,
          borderWidth: 2,
          borderRadius: i === 0 ? [4, 0, 0, 4] : i === parts.length - 1 ? [0, 4, 4, 0] : 0,
        },
        label: { show: true, position: 'inside', color: '#fff', fontSize: 11, formatter: () => `${m(x.value)} · ${pct(x.value / p.total, 0)}` },
      })),
    },
    table: {
      columns: ['Part', 'Parameters', 'Share'],
      rows: [...parts.map((x) => [x.name, x.value.toLocaleString('en-US'), pct(x.value / p.total, 1)]), ['Total', p.total.toLocaleString('en-US'), '100%']],
    },
  };
}

/** Tokens seen per training stage, on a log axis: pretraining dwarfs the rest. */
export function stageTokens(): Built {
  const t = D.post.tokens;
  const stages = [
    { name: 'Pretraining', tokens: D.arch.pretrainTokens },
    { name: 'General SFT', tokens: t['a3-sft-general'] },
    { name: 'Task SFT', tokens: t['a3-sft-tasks-e1'] },
    { name: 'DPO', tokens: t['a3-dpo'] },
  ];
  return {
    option: {
      ...BASE,
      grid: { ...GRID, top: 8, right: 56 },
      tooltip: tooltipStyle({ trigger: 'axis', axisPointer: { type: 'shadow' }, valueFormatter: (v: number) => `${m(v)} tokens` }),
      xAxis: { type: 'log', min: 1e6, ...axisStyle({ axisLabel: { color: INK.muted, formatter: (v: number) => (v >= 1e9 ? `${v / 1e9}B` : `${v / 1e6}M`) } }) },
      yAxis: { type: 'category', inverse: true, data: stages.map((s) => s.name), ...axisStyle({ splitLine: { show: false } }) },
      series: [
        {
          name: 'Tokens',
          type: 'bar',
          barWidth: 14,
          data: stages.map((s) => s.tokens),
          itemStyle: { color: SERIES[0], borderRadius: [0, 4, 4, 0] },
          label: { show: true, position: 'right', color: INK.secondary, fontSize: 11, formatter: (q: { value: number }) => (q.value >= 1e9 ? `${q.value / 1e9}B` : m(q.value)) },
        },
      ],
    },
    table: { columns: ['Stage', 'Tokens'], rows: stages.map((s) => [s.name, s.tokens.toLocaleString('en-US')]) },
  };
}

/** Pretraining training loss over the 5B tokens. */
export function pretrainLoss(): Built {
  const pts = D.pretrain.train;
  return {
    option: {
      ...BASE,
      grid: { ...GRID, top: 16 },
      tooltip: tooltipStyle({ trigger: 'axis', valueFormatter: (v: number) => v.toFixed(2) }),
      xAxis: { type: 'value', name: 'tokens (B)', nameLocation: 'middle', nameGap: 24, max: 5, ...axisStyle({ splitLine: { show: false } }) },
      yAxis: { type: 'value', min: 2, max: 11, ...axisStyle() },
      series: [{ name: 'Training loss', type: 'line', symbol: 'none', lineStyle: { color: SERIES[0], width: 2 }, data: pts }],
    },
    table: { columns: ['Tokens (B)', 'Training loss (nats/token)'], rows: pts.filter((_, i) => i % 6 === 0).map(([t, l]) => [t, l]) },
  };
}

/** Held-out bits per byte during pretraining, overall and per language. */
export function pretrainBpb(): Built {
  const v = D.pretrain.val;
  const series = [
    { name: 'Spanish', i: 2, color: SERIES[0] },
    { name: 'English', i: 3, color: SERIES[1] },
    { name: 'Both', i: 1, color: INK.secondary },
  ];
  return {
    option: {
      ...BASE,
      grid: { ...GRID, right: 48 },
      legend: { ...LEGEND, data: series.map((s) => s.name) },
      tooltip: tooltipStyle({ trigger: 'axis', valueFormatter: (x: number) => x.toFixed(3) }),
      xAxis: { type: 'value', name: 'tokens (B)', nameLocation: 'middle', nameGap: 24, max: 5, ...axisStyle({ splitLine: { show: false } }) },
      yAxis: { type: 'value', min: 0.8, ...axisStyle() },
      series: series.map((s) => ({
        name: s.name,
        type: 'line',
        symbol: 'circle',
        symbolSize: 5,
        lineStyle: { color: s.color, width: 2, type: s.name === 'Both' ? 'dashed' : 'solid' },
        itemStyle: { color: s.color },
        data: v.map((r) => [r[0], r[s.i]]),
        endLabel: { show: true, color: INK.secondary, fontSize: 11, formatter: (q: { value: number[] }) => q.value[1].toFixed(3) },
      })),
    },
    table: { columns: ['Tokens (B)', 'Both', 'Spanish', 'English'], rows: v.map((r) => [r[0], r[1], r[2], r[3]]) },
  };
}

/** The learning-rate schedule (warmup, constant, linear decay over the last 20 %). */
export function lrSchedule(): Built {
  const pts = D.pretrain.lr;
  return {
    option: {
      ...BASE,
      grid: { ...GRID, top: 16 },
      tooltip: tooltipStyle({ trigger: 'axis', valueFormatter: (v: number) => v.toExponential(1) }),
      xAxis: { type: 'value', name: 'tokens (B)', nameLocation: 'middle', nameGap: 24, max: 5, ...axisStyle({ splitLine: { show: false } }) },
      yAxis: { type: 'value', ...axisStyle({ axisLabel: { color: INK.muted, formatter: (v: number) => (v === 0 ? '0' : v.toExponential(0)) } }) },
      series: [
        {
          name: 'Learning rate',
          type: 'line',
          symbol: 'none',
          lineStyle: { color: SERIES[2], width: 2 },
          areaStyle: { color: SERIES[2], opacity: 0.12 },
          data: pts,
          markArea: {
            silent: true,
            itemStyle: { color: 'rgba(255,255,255,0.04)' },
            label: { color: INK.muted, fontSize: 11, position: 'insideTop' },
            data: [[{ name: 'decay', xAxis: 4 }, { xAxis: 5 }]],
          },
        },
      ],
    },
    table: { columns: ['Tokens (B)', 'Learning rate'], rows: pts.filter((_, i) => i % 8 === 0).map(([t, l]) => [t, l]) },
  };
}

/** Final bpb against tokens trained, on a log axis: more data keeps helping. */
export function scaling(): Built {
  const s = D.scaling;
  return {
    option: {
      ...BASE,
      grid: { ...GRID, top: 16, right: 28 },
      tooltip: tooltipStyle({ trigger: 'axis', valueFormatter: (v: number) => v.toFixed(3) }),
      xAxis: { type: 'log', min: 0.4, max: 6, name: 'tokens trained (B, log)', nameLocation: 'middle', nameGap: 24, ...axisStyle({ splitLine: { show: false } }) },
      yAxis: { type: 'value', min: 0.9, max: 1.2, ...axisStyle() },
      series: [
        {
          name: 'Held-out bpb',
          type: 'line',
          symbol: 'circle',
          symbolSize: 9,
          lineStyle: { color: SERIES[0], width: 2 },
          itemStyle: { color: SERIES[0], borderColor: INK.surface, borderWidth: 2 },
          data: s.map((p) => [p.tokensB, p.bpb]),
          label: { show: true, position: 'top', color: INK.secondary, fontSize: 11, formatter: (q: { value: number[] }) => q.value[1].toFixed(3) },
        },
      ],
    },
    table: { columns: ['Tokens (B)', 'Held-out bpb'], rows: s.map((p) => [p.tokensB, p.bpb]) },
  };
}

/** Each ablation's change in bpb against the baseline (negative = better), at 1B tokens. */
export function ablations(): Built {
  const rows = D.ablations.filter((a) => a.id !== 'a1-baseline').slice().sort((a, b) => a.delta - b.delta);
  return {
    option: {
      ...BASE,
      grid: { ...GRID, top: 8, right: 56 },
      tooltip: tooltipStyle({
        trigger: 'item',
        formatter: (q: { name: string; value: number; dataIndex: number }) =>
          `${q.name}<br>${q.value > 0 ? '+' : ''}${q.value.toFixed(4)} bpb · ${rows[q.dataIndex].tokS.toLocaleString('en-US')} tok/s`,
      }),
      xAxis: { type: 'value', ...axisStyle({ axisLabel: { color: INK.muted, formatter: (v: number) => (v > 0 ? `+${v}` : `${v}`) } }) },
      yAxis: { type: 'category', inverse: true, data: rows.map((a) => a.label), ...axisStyle({ splitLine: { show: false } }) },
      series: [
        {
          name: 'Δ bpb vs baseline',
          type: 'bar',
          barWidth: 12,
          data: rows.map((a) => ({
            value: a.delta,
            itemStyle: {
              color: a.delta < 0 ? DIVERGING.high : DIVERGING.low,
              borderRadius: a.delta < 0 ? [4, 0, 0, 4] : [0, 4, 4, 0],
              ...(a.id === 'a1b-lr2e-3' ? { borderColor: INK.primary, borderWidth: 2 } : {}),
            },
          })),
          label: {
            show: true,
            position: 'right',
            color: INK.secondary,
            fontSize: 11,
            formatter: (q: { value: number; dataIndex: number }) =>
              `${q.value > 0 ? '+' : ''}${q.value.toFixed(3)}${rows[q.dataIndex].id === 'a1b-lr2e-3' ? '  ✓ chosen' : ''}`,
          },
          markLine: { silent: true, symbol: 'none', lineStyle: { color: INK.axis, type: 'solid' }, label: { show: false }, data: [{ xAxis: 0 }] },
        },
      ],
    },
    table: {
      columns: ['Change', 'Held-out bpb', 'Δ vs baseline', 'Tokens/s'],
      rows: D.ablations.map((a) => [a.label, a.bpb, a.delta, a.tokS.toLocaleString('en-US')]),
    },
  };
}

function smallLine(points: readonly (readonly number[])[], name: string, color: string, xName: string, fmt: (v: number) => string, yMin?: number, yMax?: number): Built {
  return {
    option: {
      ...BASE,
      grid: { ...GRID, top: 16, right: 40 },
      tooltip: tooltipStyle({ trigger: 'axis', valueFormatter: fmt }),
      xAxis: { type: 'value', name: xName, nameLocation: 'middle', nameGap: 24, ...axisStyle({ splitLine: { show: false } }) },
      yAxis: { type: 'value', min: yMin, max: yMax, ...axisStyle({ axisLabel: { color: INK.muted, formatter: fmt } }) },
      series: [
        {
          name,
          type: 'line',
          symbol: 'circle',
          symbolSize: 6,
          lineStyle: { color, width: 2 },
          itemStyle: { color },
          data: points.map((p) => [p[0], p[1]]),
          endLabel: { show: true, color: INK.secondary, fontSize: 11, formatter: (q: { value: number[] }) => fmt(q.value[1]) },
        },
      ],
    },
    table: { columns: [xName, name], rows: points.map((p) => [p[0], p[1]]) },
  };
}

export const sftGeneral = () => smallLine(D.post.sftGeneral, 'Validation loss', SERIES[0], 'tokens (M)', (v) => v.toFixed(2));
export const sftTasks = () => smallLine(D.post.sftTasks, 'Validation loss', SERIES[0], 'tokens (M)', (v) => v.toFixed(2));
export const dpoLoss = () => smallLine(D.post.dpoLoss, 'Validation DPO loss', SERIES[1], 'tokens (M)', (v) => v.toFixed(2), 0.4, 0.7);
export const dpoAcc = () => smallLine(D.post.dpoAcc, 'Preference accuracy', SERIES[2], 'tokens (M)', (v) => pct(v, 0), 0.5, 1);

export interface GateRow {
  gate: string;
  task: 'Titles' | 'Show notes' | 'Relevance';
  threshold: number;
  /** "min": value must be ≥ threshold; "max": ≤ threshold. */
  dir: 'min' | 'max';
  sft: number;
  dpo: number;
  format: (v: number) => string;
}

const G = D.gates;
export const SPEARMAN_GATE = 0.8 * G['own-dpo'].qwenPassToPass;

export const GATES: GateRow[] = [
  { gate: 'Titles: no fallback to mT5', task: 'Titles', threshold: 0.1, dir: 'max', sft: G['own-sft'].titlesFallback, dpo: G['own-dpo'].titlesFallback, format: (v) => pct(v, 0) },
  { gate: 'Titles: right language', task: 'Titles', threshold: 0.95, dir: 'min', sft: G['own-sft'].titlesLang, dpo: G['own-dpo'].titlesLang, format: (v) => pct(v, 1) },
  { gate: 'Show notes: valid within 3 tries', task: 'Show notes', threshold: 0.95, dir: 'min', sft: G['own-sft'].shownotesValid, dpo: G['own-dpo'].shownotesValid, format: (v) => pct(v, 0) },
  { gate: 'Relevance: valid batches', task: 'Relevance', threshold: 1, dir: 'min', sft: G['own-sft'].relevanceValid, dpo: G['own-dpo'].relevanceValid, format: (v) => pct(v, 0) },
  {
    gate: 'Relevance: ranks like Qwen (Spearman)',
    task: 'Relevance',
    threshold: SPEARMAN_GATE,
    dir: 'min',
    sft: G['own-sft'].relevanceSpearman,
    dpo: G['own-dpo'].relevanceSpearman,
    format: (v) => v.toFixed(2),
  },
];

export const passes = (g: GateRow, v: number) => (g.dir === 'min' ? v >= g.threshold - 1e-9 : v <= g.threshold + 1e-9);

/** Rate gates (all in %): SFT and DPO bars with each gate's threshold as a tick. */
export function gateRates(): Built {
  const rows = GATES.filter((g) => g.dir === 'min' && g.threshold <= 1 && g.format(1).endsWith('%'));
  const names = rows.map((g) => g.gate);
  return {
    option: {
      ...BASE,
      grid: { ...GRID, right: 24 },
      legend: { ...LEGEND, data: ['After task SFT', 'After DPO (installed)', 'Gate'] },
      tooltip: tooltipStyle({ trigger: 'axis', axisPointer: { type: 'shadow' }, valueFormatter: (v: number) => pct(v, 1) }),
      xAxis: { type: 'value', min: 0, max: 1, ...axisStyle({ axisLabel: { color: INK.muted, formatter: (v: number) => pct(v, 0) } }) },
      yAxis: { type: 'category', inverse: true, data: names, ...axisStyle({ splitLine: { show: false } }) },
      series: [
        { name: 'After task SFT', type: 'bar', barWidth: 9, barGap: '30%', data: rows.map((g) => g.sft), itemStyle: { color: SERIES[1], borderRadius: [0, 4, 4, 0] } },
        { name: 'After DPO (installed)', type: 'bar', barWidth: 9, data: rows.map((g) => g.dpo), itemStyle: { color: SERIES[0], borderRadius: [0, 4, 4, 0] } },
        {
          name: 'Gate',
          type: 'scatter',
          symbol: 'rect',
          symbolSize: [3, 26],
          data: rows.map((g, i) => [g.threshold, i]),
          itemStyle: { color: INK.primary },
          tooltip: { valueFormatter: (v: number) => pct(v, 0) },
          z: 5,
        },
      ],
    },
    table: gateTable(rows),
  };
}

/** Relevance ranking: Spearman of our scores against Qwen's, with the gate and Qwen's own consistency. */
export function gateSpearman(): Built {
  const g = GATES[GATES.length - 1];
  const qwen = G['own-dpo'].qwenPassToPass;
  const bars = [
    { name: 'Own, after task SFT', value: g.sft, color: SERIES[1] },
    { name: 'Own, after DPO (installed)', value: g.dpo, color: SERIES[0] },
  ];
  return {
    option: {
      ...BASE,
      grid: { ...GRID, top: 24, right: 24 },
      tooltip: tooltipStyle({ trigger: 'axis', axisPointer: { type: 'shadow' }, valueFormatter: (v: number) => v.toFixed(3) }),
      xAxis: { type: 'value', min: -0.2, max: 0.6, ...axisStyle() },
      yAxis: { type: 'category', inverse: true, data: bars.map((b) => b.name), ...axisStyle({ splitLine: { show: false } }) },
      series: [
        {
          name: 'Spearman vs Qwen',
          type: 'bar',
          barWidth: 14,
          data: bars.map((b) => ({ value: b.value, itemStyle: { color: b.color, borderRadius: b.value < 0 ? [4, 0, 0, 4] : [0, 4, 4, 0] } })),
          label: { show: true, position: 'right', color: INK.secondary, fontSize: 11, formatter: (q: { value: number }) => q.value.toFixed(2) },
          markLine: {
            silent: true,
            symbol: 'none',
            label: { show: false },
            data: [
              { name: `gate ${SPEARMAN_GATE.toFixed(2)}`, xAxis: SPEARMAN_GATE, lineStyle: { color: INK.primary, type: 'dashed', width: 1.5 } },
              { name: `Qwen vs itself ${qwen.toFixed(2)}`, xAxis: qwen, lineStyle: { color: INK.muted, type: 'dotted', width: 1.5 } },
              { name: '', xAxis: 0, lineStyle: { color: INK.axis, type: 'solid', width: 1 } },
            ],
          },
        },
      ],
    },
    table: gateTable([g]),
  };
}

function gateTable(rows: GateRow[]): ChartTable {
  return {
    columns: ['Gate', 'Threshold', 'After task SFT', 'After DPO', 'Result'],
    rows: rows.map((g) => [g.gate, `${g.dir === 'min' ? '≥' : '≤'} ${g.format(g.threshold)}`, g.format(g.sft), g.format(g.dpo), passes(g, g.dpo) ? 'pass' : 'fail']),
  };
}
