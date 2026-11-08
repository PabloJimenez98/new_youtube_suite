/**
 * Pure ECharts option builders for the Analytics page (docs/decisions/0022 §5).
 * Each returns the option and the same data as a table, so values are never
 * readable only through a tooltip. Colour rules (dataviz skill):
 * - blue (OWN) is the owner's channel; every other channel is context gray;
 * - diverging charts use red ← gray → blue around "as expected";
 * - categorical colours (clusters) use the first 3 slots only.
 */
import type { EChartsCoreOption } from 'echarts/core';

import { ChartTable } from '../charts/chart-card';
import { BASE, DIVERGING, INK, OWN, SERIES, axisStyle, compact, escapeHtml, pct, tooltipStyle } from '../charts/viz';
import {
  ChannelStats,
  ChannelSummary,
  KeywordLift,
  MarketClusters,
  MarketDrivers,
  MarketGrowth,
  MarketOverview,
} from '../insights/insights';

export interface Built {
  option: EChartsCoreOption;
  table: ChartTable;
  height?: number;
}

const MARKET_ACCENT = SERIES[6]; // violet: market-wide aggregates (not a channel)
const rowHeight = (n: number) => Math.max(220, n * 22 + 60);
const own = (c: { own: boolean }) => (c.own ? OWN : INK.context);
const ownLabel = (name: string, isOwn: boolean) => (isOwn ? `${name} (you)` : name);

// ---- Market overview -----------------------------------------------------------

/** Views distribution per channel: p10 · Q1 · median · Q3 · p90, log scale. */
export function channelViews(o: MarketOverview): Built {
  const rows = [...o.channels].sort((a, b) => a.views_median - b.views_median);
  return {
    height: rowHeight(rows.length),
    option: {
      ...BASE,
      grid: { left: 8, right: 24, top: 8, bottom: 32, containLabel: true },
      tooltip: tooltipStyle({
        trigger: 'item',
        formatter: (p: { dataIndex: number }) => {
          const c = rows[p.dataIndex];
          return `<b>${escapeHtml(ownLabel(c.channel, c.own))}</b><br/>${c.videos} videos<br/>Median ${compact(c.views_median)} views<br/>Middle 50%: ${compact(c.views_q1)} – ${compact(c.views_q3)}<br/>10th–90th pct: ${compact(c.views_p10)} – ${compact(c.views_p90)}`;
        },
      }),
      xAxis: { type: 'log', name: 'Views per video (log scale)', nameLocation: 'middle', nameGap: 24, ...axisStyle({ axisLabel: { color: INK.muted, formatter: (v: number) => compact(v) } }) },
      yAxis: {
        type: 'category',
        data: rows.map((c) => ownLabel(c.channel, c.own)),
        ...axisStyle({ splitLine: { show: false }, axisLabel: { color: INK.secondary, fontSize: 11 } }),
      },
      series: [
        {
          type: 'boxplot',
          boxWidth: [6, 12],
          data: rows.map((c) => ({
            value: [c.views_p10, c.views_q1, c.views_median, c.views_q3, c.views_p90],
            itemStyle: { color: c.own ? `${OWN}55` : `${INK.context}44`, borderColor: own(c), borderWidth: c.own ? 2 : 1.5 },
          })),
        },
      ],
    },
    table: {
      columns: ['Channel', 'Videos', 'p10', 'Q1', 'Median', 'Q3', 'p90'],
      rows: [...rows].reverse().map((c) => [ownLabel(c.channel, c.own), c.videos, compact(c.views_p10), compact(c.views_q1), compact(c.views_median), compact(c.views_q3), compact(c.views_p90)]),
    },
  };
}

/** Median likes per 100 views per channel. */
export function engagement(o: MarketOverview): Built {
  const rows = o.channels.filter((c) => c.like_rate != null).sort((a, b) => (a.like_rate ?? 0) - (b.like_rate ?? 0));
  return {
    height: rowHeight(rows.length),
    option: {
      ...BASE,
      grid: { left: 8, right: 48, top: 8, bottom: 32, containLabel: true },
      tooltip: tooltipStyle({
        trigger: 'item',
        formatter: (p: { dataIndex: number }) => {
          const c = rows[p.dataIndex];
          return `<b>${escapeHtml(ownLabel(c.channel, c.own))}</b><br/>${pct(c.like_rate, 2)} likes per view<br/>${pct(c.comment_rate, 3)} comments per view`;
        },
      }),
      xAxis: { type: 'value', name: 'Likes per view (median)', nameLocation: 'middle', nameGap: 24, ...axisStyle({ axisLabel: { color: INK.muted, formatter: (v: number) => pct(v, 1) } }) },
      yAxis: { type: 'category', data: rows.map((c) => ownLabel(c.channel, c.own)), ...axisStyle({ splitLine: { show: false }, axisLabel: { color: INK.secondary, fontSize: 11 } }) },
      series: [
        {
          type: 'bar',
          barWidth: 10,
          itemStyle: { borderRadius: [0, 4, 4, 0] },
          data: rows.map((c) => ({ value: c.like_rate, itemStyle: { color: own(c) } })),
          label: { show: true, position: 'right', color: INK.secondary, fontSize: 10, formatter: (p: { dataIndex: number }) => (rows[p.dataIndex].own ? pct(rows[p.dataIndex].like_rate, 2) : '') },
        },
      ],
    },
    table: {
      columns: ['Channel', 'Likes / view', 'Comments / view', 'Uploads / week (90 d)', 'Median length (min)'],
      rows: [...rows].reverse().map((c) => [ownLabel(c.channel, c.own), pct(c.like_rate, 2), pct(c.comment_rate, 3), c.uploads_per_week_90d.toFixed(1), c.median_duration_min?.toFixed(0) ?? '–']),
    },
  };
}

/** Videos published per month across all tracked channels. */
export function monthly(o: MarketOverview): Built {
  return {
    option: {
      ...BASE,
      grid: { left: 8, right: 16, top: 16, bottom: 8, containLabel: true },
      tooltip: tooltipStyle({
        trigger: 'axis',
        axisPointer: { type: 'shadow' },
        formatter: (ps: { dataIndex: number }[]) => {
          const m = o.monthly[ps[0].dataIndex];
          return `<b>${m.month}</b><br/>${m.videos} videos published<br/>Median ${compact(m.median_views)} views`;
        },
      }),
      xAxis: { type: 'category', data: o.monthly.map((m) => m.month), ...axisStyle({ splitLine: { show: false } }) },
      yAxis: { type: 'value', name: 'Videos', ...axisStyle() },
      series: [{ type: 'bar', barMaxWidth: 18, itemStyle: { color: MARKET_ACCENT, borderRadius: [4, 4, 0, 0] }, data: o.monthly.map((m) => m.videos) }],
    },
    table: { columns: ['Month', 'Videos', 'Median views'], rows: o.monthly.map((m) => [m.month, m.videos, compact(m.median_views)]) },
  };
}

// ---- Data-science insights ------------------------------------------------------

/** Channels on the first two principal components, coloured by k-means cluster. */
export function clusters(c: Extract<MarketClusters, { available: true }>): Built {
  const label = (id: number) => `Group ${String.fromCharCode(65 + id)}`;
  return {
    height: 360,
    option: {
      ...BASE,
      grid: { left: 8, right: 24, top: 36, bottom: 32, containLabel: true },
      legend: { top: 0, textStyle: { color: INK.secondary }, data: c.clusters.map((k) => label(k.id)) },
      tooltip: tooltipStyle({
        trigger: 'item',
        formatter: (p: { data: { channel: string; own: boolean; cluster: number } }) =>
          `<b>${escapeHtml(ownLabel(p.data.channel, p.data.own))}</b><br/>${label(p.data.cluster)}`,
      }),
      xAxis: { type: 'value', name: `PC1 (${pct(c.explained_variance[0], 0)} of variance)`, nameLocation: 'middle', nameGap: 24, ...axisStyle() },
      yAxis: { type: 'value', name: `PC2 (${pct(c.explained_variance[1], 0)})`, ...axisStyle() },
      series: c.clusters.map((k) => ({
        name: label(k.id),
        type: 'scatter',
        itemStyle: { color: SERIES[k.id], borderColor: INK.surface, borderWidth: 2 },
        labelLayout: { hideOverlap: true },
        label: { show: true, position: 'right', color: INK.secondary, fontSize: 10, formatter: (p: { data: { channel: string; own: boolean } }) => ownLabel(p.data.channel, p.data.own) },
        data: c.points
          .filter((p) => p.cluster === k.id)
          .map((p) => ({ ...p, value: [p.x, p.y], symbolSize: p.own ? 18 : 11, itemStyle: p.own ? { borderColor: INK.primary, borderWidth: 2 } : undefined })),
      })),
    },
    table: {
      columns: ['Group', 'Channels', ...c.features.map((f) => f.label)],
      rows: c.clusters.map((k) => [
        label(k.id),
        c.points.filter((p) => p.cluster === k.id).map((p) => ownLabel(p.channel, p.own)).join(', '),
        ...c.features.map((f) => {
          const v = k.profile[f.key];
          return f.key.endsWith('_rate') ? pct(v, 2) : f.key === 'log_median_views' ? compact(10 ** v) : v.toFixed(2);
        }),
      ]),
    },
  };
}

/** Spearman ρ of each feature with age-adjusted views, with its 95% CI. */
export function drivers(d: Extract<MarketDrivers, { available: true }>): Built {
  const rows = d.correlations;
  const colour = (r: number) => (r >= 0 ? DIVERGING.high : DIVERGING.low);
  return {
    height: rowHeight(rows.length),
    option: {
      ...BASE,
      grid: { left: 8, right: 32, top: 8, bottom: 32, containLabel: true },
      tooltip: tooltipStyle({
        trigger: 'item',
        formatter: (p: { dataIndex: number }) => {
          const r = rows[p.dataIndex];
          return `<b>${escapeHtml(r.label)}</b><br/>ρ = ${r.rho.toFixed(3)}<br/>95% CI ${r.ci_low.toFixed(3)} to ${r.ci_high.toFixed(3)}<br/>${r.n.toLocaleString()} videos`;
        },
      }),
      xAxis: { type: 'value', name: 'Spearman ρ with age-adjusted views', nameLocation: 'middle', nameGap: 24, ...axisStyle() },
      yAxis: { type: 'category', data: rows.map((r) => r.label), ...axisStyle({ splitLine: { show: false }, axisLabel: { color: INK.secondary, fontSize: 11 } }) },
      series: [
        {
          // CI whisker: a 2px segment from ci_low to ci_high.
          type: 'custom',
          renderItem: (_: unknown, api: { value: (i: number) => number; coord: (v: number[]) => number[] }) => {
            const y = api.value(0);
            const a = api.coord([api.value(1), y]);
            const b = api.coord([api.value(2), y]);
            return { type: 'line', shape: { x1: a[0], y1: a[1], x2: b[0], y2: b[1] }, style: { stroke: INK.secondary, lineWidth: 2 } };
          },
          encode: { x: [1, 2], y: 0 },
          data: rows.map((r, i) => [i, r.ci_low, r.ci_high]),
          z: 1,
        },
        {
          type: 'scatter',
          symbolSize: 11,
          data: rows.map((r) => ({ value: [r.rho, r.label], itemStyle: { color: colour(r.rho), borderColor: INK.surface, borderWidth: 2 } })),
          markLine: { silent: true, symbol: 'none', lineStyle: { color: INK.axis, type: 'solid' }, label: { show: false }, data: [{ xAxis: 0 }] },
          z: 2,
        },
      ],
    },
    table: { columns: ['Feature', 'ρ', '95% CI', 'Videos'], rows: [...rows].reverse().map((r) => [r.label, r.rho.toFixed(3), `${r.ci_low.toFixed(3)} – ${r.ci_high.toFixed(3)}`, r.n]) },
  };
}

/** Median age-adjusted residual by publication weekday × hour (Madrid time). */
export function publishHeatmap(d: Extract<MarketDrivers, { available: true }>): Built {
  const hours = Array.from({ length: 24 }, (_, h) => `${h}:00`);
  const cells = d.heatmap.cells.filter((c) => c.median_residual != null);
  const max = Math.max(0.1, ...cells.map((c) => Math.abs(c.median_residual!)));
  return {
    height: 300,
    option: {
      ...BASE,
      grid: { left: 8, right: 16, top: 8, bottom: 56, containLabel: true },
      tooltip: tooltipStyle({
        trigger: 'item',
        formatter: (p: { value: [number, number, number, number] }) => {
          const [h, wd, r, n] = p.value;
          return `<b>${d.heatmap.weekdays[wd]} ${h}:00</b><br/>×${(10 ** r).toFixed(2)} the expected views<br/>${n} videos`;
        },
      }),
      xAxis: { type: 'category', data: hours, ...axisStyle({ splitLine: { show: false }, axisLabel: { color: INK.muted, interval: 2 } }) },
      yAxis: { type: 'category', data: d.heatmap.weekdays, inverse: true, ...axisStyle({ splitLine: { show: false } }) },
      visualMap: {
        min: -max,
        max,
        calculable: false,
        dimension: 2, // colour by the residual, not by the last dimension (the count)
        orient: 'horizontal',
        left: 'center',
        bottom: 0,
        itemHeight: 140,
        text: ['More views than expected', 'Fewer'],
        textStyle: { color: INK.muted, fontSize: 11 },
        inRange: { color: [DIVERGING.low, DIVERGING.mid, DIVERGING.high] },
      },
      series: [
        {
          type: 'heatmap',
          itemStyle: { borderColor: INK.surface, borderWidth: 2, borderRadius: 3 },
          data: cells.map((c) => [c.hour, c.weekday, c.median_residual, c.count]),
        },
      ],
    },
    table: {
      columns: ['Day', 'Hour', '× expected views', 'Videos'],
      rows: [...cells].sort((a, b) => b.median_residual! - a.median_residual!).map((c) => [d.heatmap.weekdays[c.weekday], `${c.hour}:00`, `×${(10 ** c.median_residual!).toFixed(2)}`, c.count]),
    },
  };
}

/** Views vs age (log-log): sampled videos, the binned median with its IQR band, and the fit. */
export function viewsByAge(g: Extract<MarketGrowth, { available: true }>): Built {
  const band = g.binned.map((b) => [b.age_days, b.q1, b.q3]);
  const fitLine = [1, Math.max(...g.binned.map((b) => b.age_days), 2)].map((a) => [a, 10 ** (g.fit.intercept + g.fit.slope * Math.log10(a))]);
  return {
    height: 340,
    option: {
      ...BASE,
      grid: { left: 8, right: 24, top: 56, bottom: 32, containLabel: true },
      legend: { top: 0, textStyle: { color: INK.secondary }, data: ['Other channels', 'Your channel', 'Median (all)', 'Middle 50%'] },
      tooltip: tooltipStyle({
        trigger: 'item',
        formatter: (p: { seriesName: string; data: { value?: number[]; channel?: string } | number[] }) => {
          const v = Array.isArray(p.data) ? p.data : p.data.value!;
          const who = !Array.isArray(p.data) && p.data.channel ? `<b>${escapeHtml(p.data.channel)}</b><br/>` : `<b>${p.seriesName}</b><br/>`;
          return `${who}${Math.round(v[0])} days old · ${compact(v[1])} views`;
        },
      }),
      xAxis: { type: 'log', name: 'Days since published (log)', nameLocation: 'middle', nameGap: 24, ...axisStyle() },
      yAxis: { type: 'log', name: 'Views (log)', ...axisStyle({ axisLabel: { color: INK.muted, formatter: (v: number) => compact(v) } }) },
      series: [
        {
          name: 'Other channels',
          type: 'scatter',
          symbolSize: 5,
          itemStyle: { color: INK.context, opacity: 0.45 },
          data: g.points.filter((p) => !p.own).map((p) => ({ value: [p.age_days, Math.max(p.views, 1)], channel: p.channel })),
        },
        {
          name: 'Your channel',
          type: 'scatter',
          symbolSize: 9,
          itemStyle: { color: OWN, borderColor: INK.surface, borderWidth: 1.5 },
          data: g.points.filter((p) => p.own).map((p) => ({ value: [p.age_days, Math.max(p.views, 1)], channel: p.channel })),
          z: 3,
        },
        // IQR band as one polygon: along Q3, back along Q1. (A stacked area
        // would fill down to the log axis minimum.)
        {
          name: 'Middle 50%',
          type: 'custom',
          itemStyle: { color: MARKET_ACCENT },
          silent: true,
          tooltip: { show: false },
          // The band's own points, so they count in the log axes' extent
          // (a placeholder 0 would break a log axis); drawn once, on the first.
          data: band,
          encode: { x: 0, y: [1, 2] },
          renderItem: (params: { dataIndex: number }, api: { coord: (v: number[]) => number[] }) =>
            params.dataIndex !== 0
              ? null
              : {
                  type: 'polygon',
                  shape: { points: [...band.map((b) => api.coord([b[0], b[2]])), ...[...band].reverse().map((b) => api.coord([b[0], b[1]]))] },
                  style: { fill: MARKET_ACCENT, opacity: 0.18 },
                },
          z: 1,
        },
        { name: 'Median (all)', type: 'line', symbol: 'circle', symbolSize: 6, lineStyle: { color: MARKET_ACCENT, width: 2 }, itemStyle: { color: MARKET_ACCENT }, data: g.binned.map((b) => [b.age_days, b.median]), z: 4 },
        { name: 'Fit', type: 'line', symbol: 'none', lineStyle: { color: INK.secondary, width: 1.5, type: 'dashed' }, data: fitLine, tooltip: { show: false }, z: 4 },
      ],
    },
    table: {
      columns: ['Age (days)', 'Q1', 'Median', 'Q3', 'Videos'],
      rows: g.binned.map((b) => [Math.round(b.age_days), compact(b.q1), compact(b.median), compact(b.q3), b.n]),
    },
  };
}

/** Per-channel growth exponent b in views ∝ age^b. */
export function growthSlopes(g: Extract<MarketGrowth, { available: true }>): Built {
  const rows = g.per_channel;
  return {
    height: rowHeight(rows.length),
    option: {
      ...BASE,
      grid: { left: 8, right: 40, top: 8, bottom: 32, containLabel: true },
      tooltip: tooltipStyle({
        trigger: 'item',
        formatter: (p: { dataIndex: number }) => {
          const r = rows[p.dataIndex];
          return `<b>${escapeHtml(ownLabel(r.channel, r.own))}</b><br/>views ∝ age<sup>${r.slope.toFixed(2)}</sup><br/>R² ${r.r2.toFixed(2)} · ${r.n} videos`;
        },
      }),
      xAxis: { type: 'value', name: 'Growth exponent b (views ∝ age^b)', nameLocation: 'middle', nameGap: 24, ...axisStyle() },
      yAxis: { type: 'category', data: rows.map((r) => ownLabel(r.channel, r.own)), ...axisStyle({ splitLine: { show: false }, axisLabel: { color: INK.secondary, fontSize: 11 } }) },
      series: [
        {
          type: 'bar',
          barWidth: 10,
          data: rows.map((r) => ({ value: r.slope, itemStyle: { color: own(r), borderRadius: r.slope >= 0 ? [0, 4, 4, 0] : [4, 0, 0, 4] } })),
          markLine: { silent: true, symbol: 'none', lineStyle: { color: INK.axis, type: 'solid' }, label: { show: false }, data: [{ xAxis: 0 }] },
        },
      ],
    },
    table: { columns: ['Channel', 'Exponent b', 'R²', 'Videos'], rows: [...rows].reverse().map((r) => [ownLabel(r.channel, r.own), r.slope.toFixed(3), r.r2.toFixed(2), r.n]) },
  };
}

/** Median views gained per day, by video age bucket, with the IQR as a whisker. */
export function velocity(g: Extract<MarketGrowth, { available: true }>): Built | null {
  const v = g.velocity;
  if (!v.available || !v.buckets) return null;
  const b = v.buckets;
  return {
    option: {
      ...BASE,
      grid: { left: 8, right: 16, top: 16, bottom: 8, containLabel: true },
      tooltip: tooltipStyle({
        trigger: 'item',
        formatter: (p: { dataIndex: number }) => {
          const x = b[p.dataIndex];
          return `<b>${x.label}</b><br/>Median ${compact(x.median)} views/day<br/>Middle 50%: ${compact(x.q1)} – ${compact(x.q3)}<br/>${x.n} videos`;
        },
      }),
      xAxis: { type: 'category', data: b.map((x) => x.label), ...axisStyle({ splitLine: { show: false } }) },
      yAxis: { type: 'log', name: 'Views per day', ...axisStyle({ axisLabel: { color: INK.muted, formatter: (n: number) => compact(n) } }) },
      series: [{ type: 'bar', barMaxWidth: 36, itemStyle: { color: MARKET_ACCENT, borderRadius: [4, 4, 0, 0] }, data: b.map((x) => Math.max(x.median, 0.1)) }],
    },
    table: { columns: ['Video age', 'Median views/day', 'Q1', 'Q3', 'Videos'], rows: b.map((x) => [x.label, compact(x.median), compact(x.q1), compact(x.q3), x.n]) },
  };
}

/** Title words with the biggest lift on age-adjusted views, with 95% CIs. */
export function keywordLift(positive: KeywordLift[], negative: KeywordLift[]): Built {
  const rows = [...negative.slice(0, 8)].reverse().concat([...positive.slice(0, 8)].reverse()).reverse();
  const toPct = (lift: number) => (10 ** lift - 1) * 100;
  const sorted = rows.sort((a, b) => a.lift - b.lift);
  return {
    height: rowHeight(sorted.length),
    option: {
      ...BASE,
      grid: { left: 8, right: 32, top: 8, bottom: 32, containLabel: true },
      tooltip: tooltipStyle({
        trigger: 'item',
        formatter: (p: { dataIndex: number }) => {
          const r = sorted[p.dataIndex];
          return `<b>“${escapeHtml(r.token)}”</b><br/>×${r.multiplier.toFixed(2)} the expected views<br/>95% CI ×${(10 ** r.ci_low).toFixed(2)} – ×${(10 ** r.ci_high).toFixed(2)}<br/>${r.videos} videos, ${r.channels} channels`;
        },
      }),
      xAxis: { type: 'value', name: 'Views vs expected (%)', nameLocation: 'middle', nameGap: 24, ...axisStyle({ axisLabel: { color: INK.muted, formatter: (v: number) => `${v > 0 ? '+' : ''}${v}%` } }) },
      yAxis: { type: 'category', data: sorted.map((r) => r.token), ...axisStyle({ splitLine: { show: false }, axisLabel: { color: INK.secondary, fontSize: 11 } }) },
      series: [
        {
          type: 'bar',
          barWidth: 10,
          data: sorted.map((r) => ({
            value: toPct(r.lift),
            itemStyle: { color: r.lift >= 0 ? DIVERGING.high : DIVERGING.low, borderRadius: r.lift >= 0 ? [0, 4, 4, 0] : [4, 0, 0, 4] },
          })),
          markLine: { silent: true, symbol: 'none', lineStyle: { color: INK.axis, type: 'solid' }, label: { show: false }, data: [{ xAxis: 0 }] },
        },
      ],
    },
    table: {
      columns: ['Word', '× expected views', '95% CI', 'Videos', 'Channels'],
      rows: [...sorted].reverse().map((r) => [r.token, `×${r.multiplier.toFixed(2)}`, `×${(10 ** r.ci_low).toFixed(2)} – ×${(10 ** r.ci_high).toFixed(2)}`, r.videos, r.channels]),
    },
  };
}

// ---- Your channel (Analytics + Reporting APIs) ------------------------------------

function line(dates: string[], values: (number | null)[], name: string, fmt: (v: number) => string): EChartsCoreOption {
  return {
    ...BASE,
    grid: { left: 8, right: 16, top: 16, bottom: 8, containLabel: true },
    tooltip: tooltipStyle({ trigger: 'axis', valueFormatter: (v: number) => fmt(v) }),
    xAxis: { type: 'category', data: dates, boundaryGap: false, ...axisStyle({ splitLine: { show: false } }) },
    yAxis: { type: 'value', ...axisStyle({ axisLabel: { color: INK.muted, formatter: fmt } }) },
    series: [{ name, type: 'line', symbol: 'none', lineStyle: { color: OWN, width: 2 }, areaStyle: { color: OWN, opacity: 0.12 }, data: values }],
  };
}

export function channelViewsDaily(c: ChannelSummary): Built {
  return {
    option: line(c.daily.map((d) => d.date), c.daily.map((d) => d.views), 'Views', compact),
    table: { columns: ['Date', 'Views', 'Minutes watched', 'Likes', 'Net subscribers'], rows: c.daily.map((d) => [d.date, d.views ?? '–', d.minutes_watched?.toFixed(0) ?? '–', d.likes ?? '–', d.subscribers_net ?? '–']) },
  };
}

export function channelCtr(c: ChannelSummary): Built {
  return {
    option: line(c.reach.map((d) => d.date), c.reach.map((d) => d.ctr), 'CTR', (v) => pct(v, 1)),
    table: { columns: ['Date', 'Impressions', 'CTR'], rows: c.reach.map((d) => [d.date, d.impressions ?? '–', pct(d.ctr, 2)]) },
  };
}

export function retention(c: ChannelSummary): Built {
  const pts = c.retention.points;
  return {
    option: {
      ...BASE,
      grid: { left: 8, right: 16, top: 16, bottom: 32, containLabel: true },
      tooltip: tooltipStyle({ trigger: 'axis', valueFormatter: (v: number) => pct(v, 0) }),
      xAxis: { type: 'value', min: 0, max: 1, name: 'Position in the video', nameLocation: 'middle', nameGap: 24, ...axisStyle({ axisLabel: { color: INK.muted, formatter: (v: number) => pct(v, 0) } }) },
      yAxis: { type: 'value', ...axisStyle({ axisLabel: { color: INK.muted, formatter: (v: number) => pct(v, 0) } }) },
      series: [{ name: 'Still watching', type: 'line', symbol: 'none', lineStyle: { color: OWN, width: 2 }, areaStyle: { color: OWN, opacity: 0.12 }, data: pts.map((p) => [p.ratio, p.watch_ratio]) }],
    },
    table: { columns: ['Position', 'Still watching (avg)'], rows: pts.filter((_, i) => i % 5 === 0).map((p) => [pct(p.ratio, 0), pct(p.watch_ratio, 1)]) },
  };
}

export const ownChannelOf = (rows: ChannelStats[]) => rows.find((c) => c.own) ?? null;
