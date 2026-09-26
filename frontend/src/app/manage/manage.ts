import { Component, OnDestroy, OnInit, computed, inject, signal } from '@angular/core';
import { Subscription, forkJoin, interval } from 'rxjs';

import { JobOut } from '../jobs/job.model';
import { Jobs } from '../jobs/jobs';
import { UploadOut } from '../uploads/upload.model';
import { Uploads } from '../uploads/uploads';
import { Upload } from '../uploads/upload/upload';
import { DeleteConfirmDialog } from './delete-confirm-dialog/delete-confirm-dialog';
import { VideoRow } from './video-row/video-row';

@Component({
  selector: 'app-manage',
  imports: [Upload, VideoRow, DeleteConfirmDialog],
  templateUrl: './manage.html',
  styleUrl: './manage.scss',
})
export class Manage implements OnInit, OnDestroy {
  private readonly uploads = inject(Uploads);
  private readonly jobsService = inject(Jobs);

  // Simple fixed-interval polling (per docs/decisions/0006) rather than
  // websockets -- just re-fetches jobs, not the uploads list, so an
  // in-flight run's status stays live without a full page reload.
  private static readonly JOB_POLL_INTERVAL_MS = 3000;
  private pollSubscription?: Subscription;

  protected readonly videos = signal<UploadOut[]>([]);
  protected readonly loading = signal(true);
  protected readonly errorMessage = signal<string | null>(null);
  protected readonly pendingDelete = signal<UploadOut | null>(null);
  protected readonly deleting = signal(false);

  private readonly jobsByUploadMap = signal<Map<number, JobOut[]>>(new Map());
  protected readonly jobsFor = computed(() => this.jobsByUploadMap());

  ngOnInit(): void {
    this.refresh();
    this.pollSubscription = interval(Manage.JOB_POLL_INTERVAL_MS).subscribe(() =>
      this.refreshJobs(),
    );
  }

  ngOnDestroy(): void {
    this.pollSubscription?.unsubscribe();
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

  /** Polling tick: jobs only. Failures are swallowed -- a transient
   * blip shouldn't blank out an already-loaded page; the next tick
   * just tries again. */
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
    this.refresh();
  }

  protected requestDelete(upload: UploadOut): void {
    this.pendingDelete.set(upload);
  }

  protected cancelDelete(): void {
    this.pendingDelete.set(null);
  }

  protected confirmDelete(): void {
    const target = this.pendingDelete();
    if (!target) {
      return;
    }
    this.deleting.set(true);
    this.uploads.deleteUpload(target.id).subscribe({
      next: () => {
        this.videos.update((videos) => videos.filter((v) => v.id !== target.id));
        this.pendingDelete.set(null);
        this.deleting.set(false);
      },
      error: () => {
        this.errorMessage.set('Could not delete that video.');
        this.pendingDelete.set(null);
        this.deleting.set(false);
      },
    });
  }
}
