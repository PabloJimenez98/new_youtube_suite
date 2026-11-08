import { DatePipe } from '@angular/common';
import { Component, OnDestroy, OnInit, computed, inject, signal } from '@angular/core';
import { Router, RouterLink } from '@angular/router';
import { Subscription, forkJoin, interval } from 'rxjs';

import { compact } from '../charts/viz';
import { displayName, doneCount, formatDuration, nextSuggested, toolStates } from '../explain/pipeline';
import { JobOut, TOOL_LABELS, TOOL_TYPES } from '../jobs/job.model';
import { Jobs } from '../jobs/jobs';
import { InfoButton } from '../shared/info-button/info-button';
import { UploadOut } from '../uploads/upload.model';
import { Uploads } from '../uploads/uploads';
import { Upload } from '../uploads/upload/upload';

export type SortKey = 'name' | 'published' | 'views' | 'length' | 'progress';

/**
 * Manage: a table of episodes with a preview, their YouTube numbers and
 * pipeline progress, filterable by publication date and views
 * (docs/decisions/0021, 0022 §6). The tools live on /manage/:id.
 */
@Component({
  selector: 'app-manage',
  imports: [DatePipe, InfoButton, RouterLink, Upload],
  templateUrl: './manage.html',
  styleUrl: './manage.scss',
})
export class Manage implements OnInit, OnDestroy {
  private readonly uploads = inject(Uploads);
  private readonly jobsService = inject(Jobs);
  private readonly router = inject(Router);

  // Simple fixed-interval polling (per docs/decisions/0006) rather than
  // websockets -- just re-fetches jobs, not the uploads list, so the
  // progress dots stay live without a full page reload.
  private static readonly JOB_POLL_INTERVAL_MS = 3000;
  private pollSubscription?: Subscription;

  protected readonly videos = signal<UploadOut[]>([]);
  protected readonly loading = signal(true);
  protected readonly errorMessage = signal<string | null>(null);
  protected readonly uploadOpen = signal(false);
  protected readonly toolTypes = TOOL_TYPES;
  protected readonly toolLabels = TOOL_LABELS;
  protected readonly compact = compact;

  // ---- Filters & sort --------------------------------------------------------------
  protected readonly search = signal('');
  /** yyyy-mm-dd from <input type=date>, '' = open. */
  protected readonly publishedFrom = signal('');
  protected readonly publishedTo = signal('');
  protected readonly minViews = signal<number | null>(null);
  protected readonly maxViews = signal<number | null>(null);
  protected readonly sortKey = signal<SortKey>('published');
  protected readonly sortDir = signal<'asc' | 'desc'>('desc');
  protected readonly filtersActive = computed(
    () => !!(this.search() || this.publishedFrom() || this.publishedTo() || this.minViews() != null || this.maxViews() != null),
  );

  private readonly jobsByUploadMap = signal<Map<number, JobOut[]>>(new Map());
  protected readonly jobsFor = computed(() => this.jobsByUploadMap());

  private readonly allRows = computed(() =>
    this.videos().map((video) => {
      const states = toolStates(this.jobsFor().get(video.id) ?? []);
      return {
        video,
        name: displayName(video.filename),
        duration: formatDuration(video.duration_seconds),
        published: video.youtube?.published_at ?? null,
        views: video.youtube?.view_count ?? null,
        thumbnail: video.youtube?.thumbnail_url ?? null,
        states,
        done: doneCount(states),
        next: nextSuggested(states),
        running: TOOL_TYPES.some((t) => states[t].status === 'running' || states[t].status === 'queued'),
      };
    }),
  );

  protected readonly rows = computed(() => {
    const query = this.search().trim().toLowerCase();
    const from = this.publishedFrom();
    const to = this.publishedTo();
    const min = this.minViews();
    const max = this.maxViews();
    const filtered = this.allRows().filter((r) => {
      if (query && !r.name.toLowerCase().includes(query)) return false;
      // A date or views filter excludes episodes without YouTube data.
      const day = r.published?.slice(0, 10) ?? null;
      if (from && (!day || day < from)) return false;
      if (to && (!day || day > to)) return false;
      if (min != null && (r.views == null || r.views < min)) return false;
      if (max != null && (r.views == null || r.views > max)) return false;
      return true;
    });
    const key = this.sortKey();
    const dir = this.sortDir() === 'asc' ? 1 : -1;
    const value = (r: (typeof filtered)[number]): string | number | null =>
      key === 'name' ? r.name.toLowerCase() : key === 'published' ? r.published : key === 'views' ? r.views : key === 'length' ? r.video.duration_seconds : r.done;
    return [...filtered].sort((a, b) => {
      const va = value(a);
      const vb = value(b);
      if (va == null && vb == null) return 0;
      if (va == null) return 1; // missing values always last
      if (vb == null) return -1;
      return (va < vb ? -1 : va > vb ? 1 : 0) * dir;
    });
  });

  protected readonly viewRange = computed(() => {
    const views = this.allRows().map((r) => r.views).filter((v): v is number => v != null);
    return views.length ? { min: Math.min(...views), max: Math.max(...views) } : null;
  });

  ngOnInit(): void {
    this.refresh();
    this.pollSubscription = interval(Manage.JOB_POLL_INTERVAL_MS).subscribe(() => this.refreshJobs());
  }

  ngOnDestroy(): void {
    this.pollSubscription?.unsubscribe();
  }

  protected sortBy(key: SortKey): void {
    if (this.sortKey() === key) {
      this.sortDir.set(this.sortDir() === 'asc' ? 'desc' : 'asc');
    } else {
      this.sortKey.set(key);
      this.sortDir.set(key === 'name' ? 'asc' : 'desc');
    }
  }

  protected ariaSort(key: SortKey): 'ascending' | 'descending' | 'none' {
    return this.sortKey() !== key ? 'none' : this.sortDir() === 'asc' ? 'ascending' : 'descending';
  }

  protected setNumber(target: 'min' | 'max', raw: string): void {
    const n = raw.trim() === '' ? null : Math.max(0, Number(raw));
    (target === 'min' ? this.minViews : this.maxViews).set(n == null || Number.isNaN(n) ? null : n);
  }

  protected clearFilters(): void {
    this.search.set('');
    this.publishedFrom.set('');
    this.publishedTo.set('');
    this.minViews.set(null);
    this.maxViews.set(null);
  }

  protected open(id: number): void {
    this.router.navigate(['/manage', id]);
  }

  private groupJobsByUpload(jobs: JobOut[]): Map<number, JobOut[]> {
    const grouped = new Map<number, JobOut[]>();
    for (const job of jobs) {
      const list = grouped.get(job.upload_id) ?? [];
      list.push(job);
      grouped.set(job.upload_id, list);
    }
    return grouped;
  }

  /** Polling tick: jobs only. Failures are swallowed -- the next tick tries again. */
  protected refreshJobs(): void {
    this.jobsService.listAllJobs().subscribe({
      next: (jobs) => this.jobsByUploadMap.set(this.groupJobsByUpload(jobs)),
      error: () => undefined,
    });
  }

  protected refresh(): void {
    this.loading.set(true);
    this.errorMessage.set(null);

    forkJoin({
      uploads: this.uploads.listUploads(),
      jobs: this.jobsService.listAllJobs(),
    }).subscribe({
      next: ({ uploads, jobs }) => {
        this.videos.set(uploads);
        this.jobsByUploadMap.set(this.groupJobsByUpload(jobs));
        this.loading.set(false);
      },
      error: () => {
        this.errorMessage.set('Could not load your videos.');
        this.loading.set(false);
      },
    });
  }

  protected onUploaded(): void {
    this.uploadOpen.set(false);
    this.refresh();
  }
}
