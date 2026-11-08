import { DatePipe } from '@angular/common';
import { Component, OnInit, computed, inject, signal } from '@angular/core';
import { Router, RouterLink } from '@angular/router';
import { forkJoin } from 'rxjs';

import { Chart } from '../charts/chart';
import { ChartCard, ChartHelp } from '../charts/chart-card';
import { compact } from '../charts/viz';
import { displayName, hasSucceeded } from '../explain/pipeline';
import { episodesWithStats, jobsPerDay, lengthVsViews, viewsOverTime } from './dashboard-charts';
import { JobOut, JobStatus, TOOL_LABELS, TOOL_TYPES, ToolType } from '../jobs/job.model';
import { Jobs } from '../jobs/jobs';
import { InfoButton } from '../shared/info-button/info-button';
import { Shorts } from '../shorts/shorts';
import { UploadOut } from '../uploads/upload.model';
import { Uploads } from '../uploads/uploads';
import { StatTile } from './stat-tile/stat-tile';

interface ActivityItem {
  label: string;
  at: string;
  uploadId: number;
}

const JOB_STATUSES: readonly JobStatus[] = ['queued', 'running', 'succeeded', 'failed'];

/**
 * Real numbers only (docs/decisions/0021): the M1-era dummy tiles are
 * gone. Charts of the episodes' YouTube numbers and of the job activity
 * (docs/decisions/0022 §6), plus "Pipeline coverage".
 */
@Component({
  selector: 'app-dashboard',
  imports: [Chart, ChartCard, StatTile, DatePipe, InfoButton, RouterLink],
  templateUrl: './dashboard.html',
  styleUrl: './dashboard.scss',
})
export class Dashboard implements OnInit {
  private readonly uploads = inject(Uploads);
  private readonly jobs = inject(Jobs);
  private readonly shorts = inject(Shorts);
  private readonly router = inject(Router);

  protected readonly compact = compact;
  private readonly episodes = computed(() => episodesWithStats(this.videos()));
  protected readonly linkedCount = computed(() => this.episodes().length);
  protected readonly totalViews = computed(() => this.episodes().reduce((s, e) => s + e.views, 0));
  protected readonly medianViews = computed(() => {
    const v = this.episodes().map((e) => e.views).sort((a, b) => a - b);
    return v.length ? v[Math.floor(v.length / 2)] : null;
  });
  protected readonly viewsChart = computed(() => (this.episodes().length ? viewsOverTime(this.episodes()) : null));
  protected readonly lengthChart = computed(() => (this.episodes().length > 2 ? lengthVsViews(this.episodes()) : null));
  protected readonly jobsChart = computed(() => jobsPerDay(this.allJobs()));
  protected readonly inFlight = computed(() => this.allJobs().filter((j) => j.status === 'queued' || j.status === 'running').length);
  protected readonly topEpisodes = computed(() =>
    this.videos()
      .filter((v) => v.youtube?.view_count != null)
      .sort((a, b) => b.youtube!.view_count! - a.youtube!.view_count!)
      .slice(0, 5)
      .map((v) => ({ id: v.id, name: displayName(v.filename), views: v.youtube!.view_count!, thumbnail: v.youtube!.thumbnail_url, published: v.youtube!.published_at })),
  );

  readonly help: Record<string, ChartHelp> = {
    views: {
      what: 'Views of each of your episodes, in publication order, with the median as a reference line.',
      method: 'Each episode is linked to its YouTube video through the ID in its filename; the views are the latest daily snapshot of the market pipeline.',
      read: 'Older episodes have had longer to collect views, so compare neighbours rather than the first and the last. Click a bar to open the episode.',
    },
    length: {
      what: 'Whether longer episodes get more or fewer views.',
      method: 'Episode length from the uploaded file, views from the latest YouTube snapshot.',
      read: 'A rising cloud means longer episodes do better. With a few dozen episodes this is a hint, not proof: the Analytics page tests it on the whole market.',
    },
    jobs: {
      what: 'How many tool runs finished each day over the last two weeks, and how many failed.',
      method: 'Jobs by completion day (creation day for unfinished ones).',
      read: 'A red segment is a failed run: open the episode to see the error on its tool.',
    },
  };

  protected openEpisode(datum: unknown): void {
    const id = (datum as { id?: number } | null)?.id;
    if (id != null) this.router.navigate(['/manage', id]);
  }

  protected readonly loading = signal(true);
  protected readonly errorMessage = signal<string | null>(null);

  private readonly videos = signal<UploadOut[]>([]);
  private readonly allJobs = signal<JobOut[]>([]);
  protected readonly shortsGenerated = signal(0);

  protected readonly totalVideos = computed(() => this.videos().length);

  protected readonly totalRuntimeLabel = computed(() => {
    const withDuration = this.videos().filter((v) => v.duration_seconds != null);
    const totalSeconds = withDuration.reduce((sum, v) => sum + (v.duration_seconds ?? 0), 0);
    const hours = Math.floor(totalSeconds / 3600);
    const minutes = Math.round((totalSeconds % 3600) / 60);
    const missing = this.videos().length - withDuration.length;
    return {
      value: `${hours}h ${minutes}m`,
      hint: missing > 0 ? `${missing} video(s) missing duration` : undefined,
    };
  });

  protected readonly jobsByStatus = computed(() => {
    const counts: Record<JobStatus, number> = { queued: 0, running: 0, succeeded: 0, failed: 0 };
    for (const job of this.allJobs()) counts[job.status]++;
    return JOB_STATUSES.map((status) => ({ status, count: counts[status] }));
  });

  /** Per tool: how many episodes have a result from it. */
  protected readonly coverage = computed(() => {
    const byUpload = new Map<number, JobOut[]>();
    for (const job of this.allJobs()) {
      byUpload.set(job.upload_id, [...(byUpload.get(job.upload_id) ?? []), job]);
    }
    const total = this.videos().length;
    return TOOL_TYPES.map((tool: ToolType) => {
      const done = this.videos().filter((v) => hasSucceeded(byUpload.get(v.id) ?? [], tool)).length;
      return { tool, label: TOOL_LABELS[tool], done, total, pct: total ? Math.round((done / total) * 100) : 0 };
    });
  });

  protected readonly recentActivity = computed<ActivityItem[]>(() => {
    const names = new Map(this.videos().map((v) => [v.id, displayName(v.filename)]));
    const uploadItems: ActivityItem[] = this.videos().map((v) => ({
      label: `Uploaded “${displayName(v.filename)}”`,
      at: v.uploaded_at,
      uploadId: v.id,
    }));
    const jobItems: ActivityItem[] = this.allJobs().map((j) => ({
      label: `${TOOL_LABELS[j.tool_type]} ${j.status === 'failed' ? 'failed' : j.status === 'succeeded' ? 'done' : j.status} · ${names.get(j.upload_id) ?? 'episode'}`,
      at: j.completed_at ?? j.created_at,
      uploadId: j.upload_id,
    }));
    return [...uploadItems, ...jobItems]
      .sort((a, b) => new Date(b.at).getTime() - new Date(a.at).getTime())
      .slice(0, 6);
  });

  ngOnInit(): void {
    forkJoin({
      uploads: this.uploads.listUploads(),
      jobs: this.jobs.listAllJobs(),
      clips: this.shorts.countClips(),
    }).subscribe({
      next: ({ uploads, jobs, clips }) => {
        this.videos.set(uploads);
        this.allJobs.set(jobs);
        this.shortsGenerated.set(clips);
        this.loading.set(false);
      },
      error: () => {
        this.errorMessage.set('Could not load dashboard data.');
        this.loading.set(false);
      },
    });
  }
}
