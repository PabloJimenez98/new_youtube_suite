/** ECharts option builders for the Dashboard (docs/decisions/0022 §6). */
import type { EChartsCoreOption } from 'echarts/core';

import { ChartTable } from '../charts/chart-card';
import { BASE, INK, OWN, axisStyle, compact, escapeHtml, tooltipStyle } from '../charts/viz';
import { displayName } from '../explain/pipeline';
import { JobOut } from '../jobs/job.model';
import { UploadOut } from '../uploads/upload.model';

export interface Built {
  option: EChartsCoreOption;
  table: ChartTable;
}

/** Status colours (reserved: they only ever mean job status). */
const STATUS = { succeeded: '#0ca30c', failed: '#d03b3b' };

interface Ep {
  id: number;
  name: string;
  published: string;
  views: number;
  likes: number | null;
  minutes: number | null;
}

export function episodesWithStats(uploads: UploadOut[]): Ep[] {
  return uploads
    .filter((u) => u.youtube?.view_count != null)
    .map((u) => ({
      id: u.id,
      name: displayName(u.filename),
      published: u.youtube!.published_at,
      views: u.youtube!.view_count!,
      likes: u.youtube!.like_count,
      minutes: u.duration_seconds != null ? u.duration_seconds / 60 : null,
    }))
    .sort((a, b) => a.published.localeCompare(b.published));
}

/** Views of each episode, in publication order. */
export function viewsOverTime(eps: Ep[]): Built {
  const median = eps.length ? [...eps].sort((a, b) => a.views - b.views)[Math.floor(eps.length / 2)].views : 0;
  return {
    option: {
      ...BASE,
      grid: { left: 8, right: 16, top: 24, bottom: 8, containLabel: true },
      tooltip: tooltipStyle({
        trigger: 'item',
        formatter: (p: { dataIndex: number }) => {
          const e = eps[p.dataIndex];
          return `<b>${escapeHtml(e.name)}</b><br/>${e.published.slice(0, 10)}<br/>${compact(e.views)} views${e.likes != null ? ` · ${compact(e.likes)} likes` : ''}<br/><span style="color:${INK.muted}">Click to open</span>`;
        },
      }),
      xAxis: { type: 'category', data: eps.map((e) => e.published.slice(0, 10)), ...axisStyle({ splitLine: { show: false } }) },
      yAxis: { type: 'value', ...axisStyle({ axisLabel: { color: INK.muted, formatter: (v: number) => compact(v) } }) },
      series: [
        {
          type: 'bar',
          barMaxWidth: 14,
          itemStyle: { color: OWN, borderRadius: [4, 4, 0, 0] },
          data: eps.map((e) => ({ value: e.views, id: e.id })),
          markLine: {
            silent: true,
            symbol: 'none',
            lineStyle: { color: INK.secondary, type: 'solid', width: 1 },
            label: { color: INK.secondary, formatter: `median ${compact(median)}`, position: 'insideEndTop' },
            data: [{ yAxis: median }],
          },
        },
      ],
    },
    table: { columns: ['Published', 'Episode', 'Views', 'Likes'], rows: [...eps].reverse().map((e) => [e.published.slice(0, 10), e.name, compact(e.views), compact(e.likes)]) },
  };
}

/** Each episode's length against its views. */
export function lengthVsViews(eps: Ep[]): Built {
  const pts = eps.filter((e) => e.minutes != null);
  return {
    option: {
      ...BASE,
      grid: { left: 8, right: 16, top: 16, bottom: 32, containLabel: true },
      tooltip: tooltipStyle({
        trigger: 'item',
        formatter: (p: { dataIndex: number }) => {
          const e = pts[p.dataIndex];
          return `<b>${escapeHtml(e.name)}</b><br/>${e.minutes!.toFixed(0)} min · ${compact(e.views)} views`;
        },
      }),
      xAxis: { type: 'value', name: 'Length (min)', nameLocation: 'middle', nameGap: 24, scale: true, ...axisStyle() },
      yAxis: { type: 'value', scale: true, ...axisStyle({ axisLabel: { color: INK.muted, formatter: (v: number) => compact(v) } }) },
      series: [
        {
          type: 'scatter',
          symbolSize: 10,
          itemStyle: { color: OWN, borderColor: INK.surface, borderWidth: 2 },
          data: pts.map((e) => ({ value: [e.minutes, e.views], id: e.id })),
        },
      ],
    },
    table: { columns: ['Episode', 'Length (min)', 'Views'], rows: pts.map((e) => [e.name, e.minutes!.toFixed(0), compact(e.views)]) },
  };
}

/** Finished jobs per day over the last `days` days, by outcome. */
export function jobsPerDay(jobs: JobOut[], today = new Date(), days = 14): Built {
  const keys: string[] = [];
  for (let i = days - 1; i >= 0; i--) {
    const d = new Date(today);
    d.setDate(d.getDate() - i);
    keys.push(d.toISOString().slice(0, 10));
  }
  const count = (status: 'succeeded' | 'failed') =>
    keys.map((k) => jobs.filter((j) => j.status === status && (j.completed_at ?? j.created_at).slice(0, 10) === k).length);
  const ok = count('succeeded');
  const failed = count('failed');
  return {
    option: {
      ...BASE,
      grid: { left: 8, right: 16, top: 32, bottom: 8, containLabel: true },
      legend: { top: 0, textStyle: { color: INK.secondary }, data: ['Done', 'Failed'] },
      tooltip: tooltipStyle({ trigger: 'axis', axisPointer: { type: 'shadow' } }),
      xAxis: { type: 'category', data: keys.map((k) => k.slice(5)), ...axisStyle({ splitLine: { show: false } }) },
      yAxis: { type: 'value', minInterval: 1, ...axisStyle() },
      series: [
        { name: 'Done', type: 'bar', stack: 'jobs', barMaxWidth: 18, itemStyle: { color: STATUS.succeeded, borderColor: INK.surface, borderWidth: 1 }, data: ok },
        { name: 'Failed', type: 'bar', stack: 'jobs', barMaxWidth: 18, itemStyle: { color: STATUS.failed, borderRadius: [4, 4, 0, 0], borderColor: INK.surface, borderWidth: 1 }, data: failed },
      ],
    },
    table: { columns: ['Day', 'Done', 'Failed'], rows: keys.map((k, i) => [k, ok[i], failed[i]]) },
  };
}
