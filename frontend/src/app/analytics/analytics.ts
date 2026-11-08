import { DatePipe, DecimalPipe } from '@angular/common';
import { Component, OnDestroy, OnInit, WritableSignal, computed, inject, signal } from '@angular/core';
import { Observable } from 'rxjs';

import { ChartCard, ChartHelp } from '../charts/chart-card';
import { Chart } from '../charts/chart';
import { compact, pct } from '../charts/viz';
import {
  ChannelSummary,
  Insights,
  MarketClusters,
  MarketDrivers,
  MarketFormat,
  MarketGrowth,
  MarketKeywords,
  MarketOverview,
  Stored,
} from '../insights/insights';
import { InfoButton } from '../shared/info-button/info-button';
import * as build from './analytics-charts';
import { YoutubeConnection } from './youtube-connection/youtube-connection';

/** One request's state: loading → data or error. */
interface Slot<T> {
  loading: boolean;
  data: T | null;
  error: boolean;
}

const idle = <T>(): Slot<T> => ({ loading: true, data: null, error: false });

/**
 * Analytics (docs/decisions/0022): the connected channel's private data,
 * the tracked market's public data, and data-science analyses of the
 * market. Every chart has an explanation tooltip and a table view.
 */
@Component({
  selector: 'app-analytics',
  imports: [Chart, ChartCard, DatePipe, DecimalPipe, InfoButton, YoutubeConnection],
  templateUrl: './analytics.html',
  styleUrl: './analytics.scss',
})
export class Analytics implements OnInit, OnDestroy {
  private readonly insights = inject(Insights);

  protected readonly formats: { id: MarketFormat; label: string }[] = [
    { id: 'long', label: 'Full episodes (> 3 min)' },
    { id: 'short', label: 'Shorts & clips (≤ 3 min)' },
    { id: 'all', label: 'All videos' },
  ];
  protected readonly format = signal<MarketFormat>('long');

  protected readonly channel = signal<Slot<ChannelSummary>>(idle());
  protected readonly overview = signal<Slot<MarketOverview>>(idle());
  protected readonly clusters = signal<Slot<MarketClusters>>(idle());
  protected readonly drivers = signal<Slot<MarketDrivers>>(idle());
  protected readonly growth = signal<Slot<MarketGrowth>>(idle());
  protected readonly keywords = signal<Slot<MarketKeywords>>(idle());

  // ---- Built charts (null = not loaded / not available) --------------------------
  protected readonly views = computed(() => this.when(this.overview(), build.channelViews));
  protected readonly likes = computed(() => this.when(this.overview(), build.engagement));
  protected readonly months = computed(() => this.when(this.overview(), build.monthly));
  protected readonly clusterChart = computed(() => {
    const c = this.clusters().data;
    return c?.available ? build.clusters(c) : null;
  });
  protected readonly driverChart = computed(() => {
    const d = this.drivers().data;
    return d?.available ? build.drivers(d) : null;
  });
  protected readonly heatmap = computed(() => {
    const d = this.drivers().data;
    return d?.available ? build.publishHeatmap(d) : null;
  });
  protected readonly ageChart = computed(() => {
    const g = this.growth().data;
    return g?.available ? build.viewsByAge(g) : null;
  });
  protected readonly slopes = computed(() => {
    const g = this.growth().data;
    return g?.available ? build.growthSlopes(g) : null;
  });
  protected readonly velocityChart = computed(() => {
    const g = this.growth().data;
    return g?.available ? build.velocity(g) : null;
  });
  protected readonly velocityReason = computed(() => {
    const g = this.growth().data;
    if (!g?.available) return 'Not enough data yet.';
    return `${g.velocity.reason ?? ''} The daily market pull adds a snapshot every day: this chart fills in after the next pulls.`;
  });
  protected readonly keywordChart = computed(() => {
    const k = this.keywords().data;
    return k?.available ? build.keywordLift(k.positive, k.negative) : null;
  });
  protected readonly channelViews = computed(() => {
    const c = this.channel().data;
    return c?.daily.length ? build.channelViewsDaily(c) : null;
  });
  protected readonly channelCtr = computed(() => {
    const c = this.channel().data;
    return c?.reach.length ? build.channelCtr(c) : null;
  });
  protected readonly retention = computed(() => {
    const c = this.channel().data;
    return c?.retention.points.length ? build.retention(c) : null;
  });

  // ---- Headline numbers --------------------------------------------------------------
  protected readonly ownChannel = computed(() => build.ownChannelOf(this.overview().data?.channels ?? []));
  protected readonly ownRank = computed(() => {
    const rows = [...(this.overview().data?.channels ?? [])].sort((a, b) => b.views_median - a.views_median);
    const i = rows.findIndex((c) => c.own);
    return i < 0 ? null : { rank: i + 1, of: rows.length };
  });
  protected readonly clusterInfo = computed(() => {
    const c = this.clusters().data;
    return c?.available ? c : null;
  });
  protected readonly topDriver = computed(() => {
    const d = this.drivers().data;
    if (!d?.available || !d.correlations.length) return null;
    return [...d.correlations].sort((a, b) => Math.abs(b.rho) - Math.abs(a.rho))[0];
  });
  protected readonly compact = compact;
  protected readonly pct = pct;

  readonly help: Record<string, ChartHelp> = {
    views: {
      what: 'How many views each tracked channel’s videos get: the box is the middle 50% of its videos, the line inside is the median, the whiskers reach the 10th and 90th percentiles.',
      method: 'Latest daily snapshot of every video from the YouTube Data API, filtered by the format above. Log scale, because channels differ by orders of magnitude.',
      read: 'A long box means uneven performance (a few hits, many quiet videos). Your channel is the blue one. Older videos have had more time to collect views, so this is a lifetime comparison, not a speed one.',
    },
    likes: {
      what: 'The median share of viewers who like a video, per channel.',
      method: 'likes ÷ views per video, then the channel median (a median, so a few viral videos don’t dominate).',
      read: 'A proxy for how engaged the audience is. Compare channels of similar size: small, loyal audiences usually like more per view.',
    },
    months: {
      what: 'How many videos the tracked channels published each month.',
      method: 'Counts by publication month (UTC) across all tracked channels, last 24 months.',
      read: 'The market’s output over time. The current month is partial.',
    },
    clusters: {
      what: 'Which channels resemble each other, on six features at once: median views, likes and comments per view, uploads per week, median length and how uneven their views are.',
      method: 'Features standardised (z-scores), then k-means clustering with k chosen by the best silhouette score (k ≤ 3), then PCA to project the six dimensions onto the two that keep the most variance.',
      read: 'Nearby dots are similar channels; colours are the k-means groups. The axes are combinations of features, not single metrics: the table shows each group’s profile. A silhouette under 0.25 means the groups are weak tendencies, not clear-cut types.',
    },
    drivers: {
      what: 'How strongly each video or title trait goes with getting more views than expected.',
      method: 'Each video’s views are first adjusted for its channel and age: per channel, a regression log(views) ~ log(age) gives the expected views, and the residual is the surprise. Then a Spearman rank correlation per trait, with a 95% bootstrap confidence interval.',
      read: 'ρ > 0: videos with more of the trait tend to beat expectations. ρ near 0 (|ρ| < 0.1) is a negligible link. This is correlation across many channels, not causation: e.g. long videos may do better because of what they are about.',
    },
    heatmap: {
      what: 'Whether the publication day and hour go with more or fewer views than expected.',
      method: 'Median age- and channel-adjusted residual of the videos published in each weekday × hour slot (Madrid time). Slots with fewer than 15 videos are left blank.',
      read: 'Blue slots beat expectations, red ones trail them, gray is as expected. Each channel publishes at its own habitual times, so a slot partly reflects which channels use it.',
    },
    age: {
      what: 'How views accumulate with a video’s age, across the market.',
      method: 'All of your channel’s videos (blue) and a random sample of 2,000 from the other channels (gray), the median views in log-spaced age bins (violet) with the middle 50% band, and a log-log least-squares fit over every video (dashed).',
      read: 'This is a snapshot of different videos at different ages, not one video followed over time: old videos come from when channels were smaller, which flattens the curve. The per-channel exponents beside it are fairer.',
    },
    slopes: {
      what: 'For each channel, how fast views grow with age: views ∝ age^b.',
      method: 'Per-channel least-squares fit of log(views) on log(age), using channels with at least 20 videos.',
      read: 'b > 0: older videos keep collecting views (evergreen catalogue). b ≤ 0: recent videos outperform old ones, typically a growing channel.',
    },
    velocity: {
      what: 'How many views a video gains per day, by how old it is.',
      method: 'Difference between each video’s oldest and newest snapshot in the last 14 days, divided by the days between them. The bars are medians.',
      read: 'Shows how fast attention decays after publication. It needs at least two daily snapshots of the same videos.',
    },
    keywords: {
      what: 'Title words that go with more (blue) or fewer (red) views than expected.',
      method: 'Candidate words are the titles’ highest summed TF-IDF terms that appear in ≥ 40 videos across ≥ 3 channels (channel names and one channel’s vocabulary are excluded). For each, mean age-adjusted residual with the word minus without it, with a 95% CI.',
      read: '+100% means videos with the word got about twice the views expected for their channel and age. Many top words are guests or topics: they describe what drew audiences, not words to paste into a title.',
    },
    channelViews: {
      what: 'Daily views of your connected channel.',
      method: 'YouTube Analytics API v2, one row per video per day, summed per day. The last 7 days are re-read each run.',
      read: 'Analytics lags 48–72 h, so the last days are incomplete.',
    },
    channelCtr: {
      what: 'Your channel’s real thumbnail click-through rate per day.',
      method: 'YouTube Reporting API daily reach reports: clicks ÷ thumbnail impressions, weighted by impressions across videos.',
      read: 'The real CTR the CTR score (heuristic v0) will be calibrated against in M15.',
    },
    retention: {
      what: 'The average share of viewers still watching at each point of your videos.',
      method: 'Analytics API retention curves (100 points per video, latest pull), averaged across videos.',
      read: 'Steep early drops mean weak openings: that’s what the hook score tries to predict.',
    },
  };

  /** Some result isn't stored yet: the worker is computing it (docs/decisions/0023). */
  protected readonly pending = computed(() =>
    [this.channel(), this.overview(), this.clusters(), this.drivers(), this.growth(), this.keywords()].some(
      (s) => (s.data as Stored | null)?.pending,
    ),
  );
  /** Loading, or not stored yet (the worker is computing it). */
  protected waiting(slot: Slot<unknown>): boolean {
    return slot.loading || !!(slot.data as Stored | null)?.pending;
  }

  /** When the market results shown were computed. */
  protected readonly computedAt = computed(() => this.overview().data?.computed_at ?? null);

  static readonly PENDING_RETRY_MS = 5000;
  static readonly PENDING_MAX_TRIES = 24; // two minutes
  private readonly timers = new Set<ReturnType<typeof setTimeout>>();
  private generation = 0;

  ngOnInit(): void {
    this.load(this.channel, () => this.insights.channel());
    this.loadMarket();
  }

  ngOnDestroy(): void {
    this.timers.forEach(clearTimeout);
  }

  protected setFormat(format: MarketFormat): void {
    if (format === this.format()) return;
    this.format.set(format);
    this.loadMarket();
  }

  private loadMarket(): void {
    const f = this.format();
    this.generation++; // retries for the previous format stop
    this.load(this.overview, () => this.insights.overview(f));
    this.load(this.clusters, () => this.insights.clusters(f));
    this.load(this.drivers, () => this.insights.drivers(f));
    this.load(this.growth, () => this.insights.growth(f));
    this.load(this.keywords, () => this.insights.keywords(f));
  }

  /**
   * Fetch into a slot. The API only reads precomputed rows, so it answers
   * at once; a `pending` answer (nothing stored yet) is retried every few
   * seconds while the worker computes it.
   */
  private load<T>(slot: WritableSignal<Slot<T>>, fetch: () => Observable<T>, tries = 0): void {
    const generation = this.generation;
    // Keep showing the previous data while refetching (no skeleton flash).
    slot.update((v) => ({ ...v, loading: true, error: false }));
    fetch().subscribe({
      next: (data) => {
        if (generation !== this.generation && slot !== this.channel) return; // a newer format won
        slot.set({ loading: false, data, error: false });
        if ((data as Stored).pending && tries < Analytics.PENDING_MAX_TRIES) {
          const timer = setTimeout(() => {
            this.timers.delete(timer);
            if (generation === this.generation || slot === this.channel) this.load(slot, fetch, tries + 1);
          }, Analytics.PENDING_RETRY_MS);
          this.timers.add(timer);
        }
      },
      error: () => slot.set({ loading: false, data: null, error: true }),
    });
  }

  private when<T, R>(slot: Slot<T>, fn: (d: T) => R): R | null {
    return slot.data ? fn(slot.data) : null;
  }
}
