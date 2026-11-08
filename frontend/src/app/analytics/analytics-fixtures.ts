/** Test fixtures shaped like the /insights responses (docs/decisions/0022). */
import {
  ChannelStats,
  ChannelSummary,
  MarketClusters,
  MarketDrivers,
  MarketGrowth,
  MarketKeywords,
  MarketOverview,
} from '../insights/insights';

const meta = { snapshot_date: '2026-11-08', videos: 300, own_channel: 'Ours' };

const channel = (id: number, name: string, median: number, own = false): ChannelStats => ({
  channel_id: id,
  channel: name,
  own,
  videos: 100,
  views_p10: median / 10,
  views_q1: median / 2,
  views_median: median,
  views_q3: median * 2,
  views_p90: median * 5,
  like_rate: 0.01 * id,
  comment_rate: 0.001,
  uploads_per_week_90d: 1.5,
  median_duration_min: 60,
});

export const OVERVIEW: MarketOverview = {
  ...meta,
  channels: [channel(1, 'Ours', 50_000, true), channel(2, 'Big', 1_000_000), channel(3, 'Small', 5_000)],
  monthly: [
    { month: '2026-10', videos: 40, median_views: 20_000 },
    { month: '2026-11', videos: 8, median_views: 9_000 },
  ],
};

export const CLUSTERS: MarketClusters = {
  ...meta,
  available: true,
  k: 2,
  silhouette: 0.31,
  explained_variance: [0.5, 0.3],
  features: [
    { key: 'log_median_views', label: 'Median views (log)' },
    { key: 'like_rate', label: 'Likes per view' },
  ],
  loadings: [],
  points: [
    { channel: 'Ours', own: true, x: 0.1, y: 0.2, cluster: 0 },
    { channel: 'Big', own: false, x: 2, y: -1, cluster: 1 },
    { channel: 'Small', own: false, x: -0.3, y: 0.4, cluster: 0 },
  ],
  clusters: [
    { id: 0, size: 2, profile: { log_median_views: 4.2, like_rate: 0.02 } },
    { id: 1, size: 1, profile: { log_median_views: 6, like_rate: 0.01 } },
  ],
};

export const DRIVERS: MarketDrivers = {
  ...meta,
  available: true,
  n: 300,
  correlations: [
    { feature: 'title_question', label: 'Title is a question', rho: -0.06, ci_low: -0.07, ci_high: -0.05, n: 300 },
    { feature: 'duration_min', label: 'Video length', rho: 0.3, ci_low: 0.28, ci_high: 0.32, n: 300 },
  ],
  heatmap: {
    weekdays: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
    cells: [
      { weekday: 0, hour: 18, count: 40, median_residual: 0.2 },
      { weekday: 6, hour: 9, count: 20, median_residual: -0.1 },
      { weekday: 3, hour: 3, count: 2, median_residual: null },
    ],
  },
};

export const GROWTH: MarketGrowth = {
  ...meta,
  available: true,
  fit: { intercept: 4, slope: 0.2, r2: 0.1, n: 300 },
  binned: [
    { age_days: 3, q1: 1000, median: 3000, q3: 9000, n: 40 },
    { age_days: 100, q1: 5000, median: 15000, q3: 40000, n: 100 },
  ],
  points: [
    { age_days: 10, views: 2000, channel: 'Ours', own: true },
    { age_days: 400, views: 90000, channel: 'Big', own: false },
  ],
  per_channel: [
    { channel: 'Big', own: false, slope: -0.1, r2: 0.05, n: 100 },
    { channel: 'Ours', own: true, slope: 0.4, r2: 0.3, n: 100 },
  ],
  velocity: { available: false, days: 0, reason: 'Velocity needs at least two daily snapshots of the same videos.' },
};

export const KEYWORDS: MarketKeywords = {
  ...meta,
  available: true,
  n: 300,
  positive: [{ token: 'aliens', videos: 55, channels: 4, lift: 0.5, ci_low: 0.3, ci_high: 0.7, multiplier: 3.16 }],
  negative: [{ token: 'covid', videos: 60, channels: 5, lift: -0.3, ci_low: -0.4, ci_high: -0.2, multiplier: 0.5 }],
};

export const CHANNEL_EMPTY: ChannelSummary = { connected: false, videos: 0, daily: [], reach: [], retention: { videos: 0, points: [] } };

export const CHANNEL_DATA: ChannelSummary = {
  connected: true,
  videos: 3,
  daily: [{ date: '2026-11-07', views: 120, minutes_watched: 300, likes: 5, comments: 1, subscribers_net: 2 }],
  reach: [{ date: '2026-11-07', impressions: 4000, ctr: 0.051 }],
  retention: { videos: 2, points: [{ ratio: 0, watch_ratio: 1 }, { ratio: 0.5, watch_ratio: 0.4 }] },
};
