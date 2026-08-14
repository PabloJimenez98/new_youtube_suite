import { DatePipe } from '@angular/common';
import { Component, OnInit, computed, inject, signal } from '@angular/core';
import { forkJoin } from 'rxjs';

import { JobOut, JobStatus, TOOL_LABELS } from '../jobs/job.model';
import { Jobs } from '../jobs/jobs';
import { UploadOut } from '../uploads/upload.model';
import { Uploads } from '../uploads/uploads';
import { Sparkline } from '../shared/sparkline/sparkline';
import { StatTile } from './stat-tile/stat-tile';
import {
  DUMMY_AVG_PREDICTED_CTR,
  DUMMY_RELEVANCE_SPARKLINE,
  DUMMY_SHORTS_GENERATED,
  DUMMY_TOP_HOOK_CLIP,
} from './dummy-stats';

interface ActivityItem {
  label: string;
  at: string;
}

const JOB_STATUSES: readonly JobStatus[] = ['queued', 'running', 'succeeded', 'failed'];

@Component({
  selector: 'app-dashboard',
  imports: [StatTile, Sparkline, DatePipe],
  templateUrl: './dashboard.html',
  styleUrl: './dashboard.scss',
})
export class Dashboard implements OnInit {
  private readonly uploads = inject(Uploads);
  private readonly jobs = inject(Jobs);

  protected readonly loading = signal(true);
  protected readonly errorMessage = signal<string | null>(null);

  private readonly videos = signal<UploadOut[]>([]);
  private readonly allJobs = signal<JobOut[]>([]);

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
    const counts: Record<JobStatus, number> = {
      queued: 0,
      running: 0,
      succeeded: 0,
      failed: 0,
    };
    for (const job of this.allJobs()) {
      counts[job.status]++;
    }
    return JOB_STATUSES.map((status) => ({ status, count: counts[status] }));
  });

  protected readonly recentActivity = computed<ActivityItem[]>(() => {
    const uploadItems: ActivityItem[] = this.videos().map((v) => ({
      label: `Uploaded "${v.filename}"`,
      at: v.uploaded_at,
    }));
    const jobItems: ActivityItem[] = this.allJobs().map((j) => ({
      label: `${TOOL_LABELS[j.tool_type]} requested`,
      at: j.created_at,
    }));
    return [...uploadItems, ...jobItems]
      .sort((a, b) => new Date(b.at).getTime() - new Date(a.at).getTime())
      .slice(0, 5);
  });

  // ---- Dummy/preview tiles -- see dashboard/dummy-stats.ts ----
  protected readonly dummyAvgCtr = DUMMY_AVG_PREDICTED_CTR;
  protected readonly dummyTopHookClip = DUMMY_TOP_HOOK_CLIP;
  protected readonly dummyRelevanceSparkline = DUMMY_RELEVANCE_SPARKLINE;
  protected readonly dummyShortsGenerated = DUMMY_SHORTS_GENERATED;

  ngOnInit(): void {
    forkJoin({
      uploads: this.uploads.listUploads(),
      jobs: this.jobs.listAllJobs(),
    }).subscribe({
      next: ({ uploads, jobs }) => {
        this.videos.set(uploads);
        this.allJobs.set(jobs);
        this.loading.set(false);
      },
      error: () => {
        this.errorMessage.set('Could not load dashboard data.');
        this.loading.set(false);
      },
    });
  }
}
