/**
 * PLACEHOLDER DATA -- entire page previews the M10+ market-analysis
 * pipeline (public curated-channel trends + the operator's own
 * connected-channel analytics). Nothing here is wired to any backend
 * endpoint. See docs/decisions/0004-full-app-shell-pivot.md.
 */
export interface MarketTrend {
  channel: string;
  trendScore: number;
}

export const DUMMY_MARKET_TRENDS: readonly MarketTrend[] = [
  { channel: 'Longform Interviews', trendScore: 82 },
  { channel: 'Tech Explainers', trendScore: 74 },
  { channel: 'News Recaps', trendScore: 61 },
  { channel: 'Comedy Panels', trendScore: 55 },
];

export const DUMMY_CHANNEL_METRICS = {
  impressions: 128_400,
  avgCtr: '5.1%',
  avgRetention: '46%',
};
