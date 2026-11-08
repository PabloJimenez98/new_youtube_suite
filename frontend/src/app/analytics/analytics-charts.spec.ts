import { INK, OWN } from '../charts/viz';
import * as build from './analytics-charts';
import { CHANNEL_DATA, CLUSTERS, DRIVERS, GROWTH, KEYWORDS, OVERVIEW } from './analytics-fixtures';

type Any = any; // eslint-disable-line @typescript-eslint/no-explicit-any -- loosely typed ECharts options
const series = (o: Any, i = 0): Any => (o['series'] as Any[])[i];

describe('analytics chart builders', () => {
  it('channel views: sorted by median, own channel in blue, the rest context gray, log axis', () => {
    const b = build.channelViews(OVERVIEW);
    const o = b.option as Any;
    expect(o['xAxis'].type).toBe('log');
    expect(o['yAxis'].data).toEqual(['Small', 'Ours (you)', 'Big']);
    const colours = series(o)['data'].map((d: Any) => d.itemStyle.borderColor);
    expect(colours).toEqual([INK.context, OWN, INK.context]);
    expect(series(o)['data'][1].value).toEqual([5000, 25000, 50000, 100000, 250000]);
    expect(b.table.rows[0][0]).toBe('Big');
  });

  it('engagement: one value per channel, only the owner’s labelled', () => {
    const o = build.engagement(OVERVIEW).option as Any;
    const fmt = series(o)['label'].formatter;
    expect(series(o)['data'].length).toBe(3);
    expect([0, 1, 2].map((i) => fmt({ dataIndex: i }))).toEqual(['1.00%', '', '']);
  });

  it('clusters: one series per group with at most 3 colours, the owner enlarged', () => {
    const b = build.clusters(CLUSTERS as Extract<typeof CLUSTERS, { available: true }>);
    const s = (b.option as Any)['series'] as Any[];
    expect(s.map((x) => x['name'])).toEqual(['Group A', 'Group B']);
    const ours = s[0]['data'].find((d: Any) => d.own);
    expect(ours.symbolSize).toBeGreaterThan(s[0]['data'].find((d: Any) => !d.own).symbolSize);
    expect(b.table.rows[0][1]).toBe('Ours (you), Small');
  });

  it('drivers: a CI whisker per feature and a zero reference line', () => {
    const b = build.drivers(DRIVERS as Extract<typeof DRIVERS, { available: true }>);
    const o = b.option as Any;
    expect(series(o, 0)['data']).toEqual([
      [0, -0.07, -0.05],
      [1, 0.28, 0.32],
    ]);
    expect(series(o, 1)['markLine'].data).toEqual([{ xAxis: 0 }]);
    expect(b.table.rows[0]).toEqual(['Video length', '0.300', '0.280 – 0.320', 300]);
  });

  it('heatmap: symmetric diverging scale, sparse slots left blank', () => {
    const b = build.publishHeatmap(DRIVERS as Extract<typeof DRIVERS, { available: true }>);
    const o = b.option as Any;
    expect(o['visualMap'].min).toBe(-o['visualMap'].max);
    expect(o['visualMap'].dimension).toBe(2); // colour by residual, not by the count
    expect(series(o)['data'].length).toBe(2);
    expect(b.table.rows[0]).toEqual(['Mon', '18:00', '×1.58', 40]);
  });

  it('views by age: own points separate from context points, IQR band as a polygon', () => {
    const o = build.viewsByAge(GROWTH as Extract<typeof GROWTH, { available: true }>).option as Any;
    const names = (o['series'] as Any[]).map((s) => s['name']);
    expect(names).toEqual(['Other channels', 'Your channel', 'Middle 50%', 'Median (all)', 'Fit']);
    // The band is one polygon: along Q3, then back along Q1.
    const poly = series(o, 2)['renderItem']({ dataIndex: 0 }, { coord: (v: number[]) => v });
    expect(series(o, 2)['renderItem']({ dataIndex: 1 }, { coord: (v: number[]) => v })).toBeNull();
    expect(poly.shape.points).toEqual([
      [3, 9000],
      [100, 40000],
      [100, 5000],
      [3, 1000],
    ]);
  });

  it('velocity returns null until there are two snapshots', () => {
    expect(build.velocity(GROWTH as Extract<typeof GROWTH, { available: true }>)).toBeNull();
  });

  it('keywords: percent vs expected, blue above and red below zero', () => {
    const b = build.keywordLift(KEYWORDS.positive, KEYWORDS.negative);
    const data = series(b.option as Any)['data'];
    expect(data.map((d: Any) => Math.round(d.value))).toEqual([-50, 216]);
    expect(b.table.rows[0][0]).toBe('aliens');
  });

  it('own-channel charts take their rows from the summary', () => {
    expect(build.channelViewsDaily(CHANNEL_DATA).table.rows).toEqual([['2026-11-07', 120, '300', 5, 2]]);
    expect(build.channelCtr(CHANNEL_DATA).table.rows).toEqual([['2026-11-07', 4000, '5.10%']]);
    expect(build.retention(CHANNEL_DATA).table.rows[0]).toEqual(['0%', '100.0%']);
  });
});
