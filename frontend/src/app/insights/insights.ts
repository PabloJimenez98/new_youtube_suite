import { HttpClient, HttpParams } from '@angular/common/http';
import { Service, inject } from '@angular/core';
import { Observable } from 'rxjs';

/** Mirrors backend/app/insights/router.py (docs/decisions/0022). */
export type MarketFormat = 'long' | 'short' | 'all';

/** Every /insights response is read from precomputed rows (docs/decisions/0023). */
export interface Stored {
  /** No stored result yet: the worker is computing it. */
  pending?: boolean;
  /** When the worker computed it (ISO), null while pending. */
  computed_at?: string | null;
  reason?: string;
}

export interface MarketMeta extends Stored {
  snapshot_date: string | null;
  videos: number;
  own_channel: string | null;
}

export interface ChannelStats {
  channel_id: number;
  channel: string;
  own: boolean;
  videos: number;
  views_p10: number;
  views_q1: number;
  views_median: number;
  views_q3: number;
  views_p90: number;
  like_rate: number | null;
  comment_rate: number | null;
  uploads_per_week_90d: number;
  median_duration_min: number | null;
}

export interface MarketOverview extends MarketMeta {
  channels: ChannelStats[];
  monthly: { month: string; videos: number; median_views: number | null }[];
}

interface Unavailable {
  available: false;
  reason: string;
}

export type MarketClusters = MarketMeta &
  (
    | Unavailable
    | {
        available: true;
        k: number;
        silhouette: number;
        explained_variance: [number, number];
        features: { key: string; label: string }[];
        loadings: { key: string; pc1: number; pc2: number }[];
        points: { channel: string; own: boolean; x: number; y: number; cluster: number }[];
        clusters: { id: number; size: number; profile: Record<string, number> }[];
      }
  );

export interface Correlation {
  feature: string;
  label: string;
  rho: number;
  ci_low: number;
  ci_high: number;
  n: number;
}

export interface HeatCell {
  weekday: number;
  hour: number;
  count: number;
  median_residual: number | null;
}

export type MarketDrivers = MarketMeta &
  (Unavailable | { available: true; n: number; correlations: Correlation[]; heatmap: { weekdays: string[]; cells: HeatCell[] } });

export interface Velocity {
  available: boolean;
  days: number;
  reason?: string;
  buckets?: { label: string; q1: number; median: number; q3: number; n: number }[];
  top?: { title: string; channel: string; views_per_day: number; age_days: number }[];
}

export type MarketGrowth = MarketMeta &
  (
    | Unavailable
    | {
        available: true;
        fit: { intercept: number; slope: number; r2: number; n: number };
        binned: { age_days: number; q1: number; median: number; q3: number; n: number }[];
        points: { age_days: number; views: number; channel: string; own: boolean }[];
        per_channel: { channel: string; own: boolean; slope: number; r2: number; n: number }[];
        velocity: Velocity;
      }
  );

export interface KeywordLift {
  token: string;
  videos: number;
  channels: number;
  lift: number;
  ci_low: number;
  ci_high: number;
  multiplier: number;
}

export type MarketKeywords = MarketMeta & { available: boolean; reason?: string; n?: number; positive: KeywordLift[]; negative: KeywordLift[] };

export interface ChannelSummary extends Stored {
  connected: boolean;
  videos: number;
  daily: { date: string; views: number | null; minutes_watched: number | null; likes: number | null; comments: number | null; subscribers_net: number | null }[];
  reach: { date: string; impressions: number | null; ctr: number | null }[];
  retention: { videos: number; points: { ratio: number; watch_ratio: number | null }[] };
}

@Service()
export class Insights {
  private readonly http = inject(HttpClient);

  private market<T>(path: string, format: MarketFormat): Observable<T> {
    return this.http.get<T>(`/insights/market/${path}`, { params: new HttpParams().set('format', format) });
  }

  overview(format: MarketFormat): Observable<MarketOverview> {
    return this.market('overview', format);
  }

  clusters(format: MarketFormat): Observable<MarketClusters> {
    return this.market('clusters', format);
  }

  drivers(format: MarketFormat): Observable<MarketDrivers> {
    return this.market('drivers', format);
  }

  growth(format: MarketFormat): Observable<MarketGrowth> {
    return this.market('growth', format);
  }

  keywords(format: MarketFormat): Observable<MarketKeywords> {
    return this.market('keywords', format);
  }

  channel(): Observable<ChannelSummary> {
    return this.http.get<ChannelSummary>('/insights/channel');
  }
}
