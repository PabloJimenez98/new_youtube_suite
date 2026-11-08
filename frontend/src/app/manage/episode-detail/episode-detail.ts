import { DatePipe } from '@angular/common';
import { Component, OnDestroy, OnInit, computed, inject, input, signal } from '@angular/core';
import { Router, RouterLink } from '@angular/router';
import { Subscription, interval } from 'rxjs';

import { compact } from '../../charts/viz';
import { displayName, formatDuration } from '../../explain/pipeline';
import { JobOut } from '../../jobs/job.model';
import { Jobs } from '../../jobs/jobs';
import { UploadOut } from '../../uploads/upload.model';
import { Uploads } from '../../uploads/uploads';
import { DeleteConfirmDialog } from '../delete-confirm-dialog/delete-confirm-dialog';
import { EpisodeWorkspace } from '../episode-workspace/episode-workspace';

/** /manage/:id -- one episode and its tools (docs/decisions/0021). */
@Component({
  selector: 'app-episode-detail',
  imports: [DatePipe, DeleteConfirmDialog, EpisodeWorkspace, RouterLink],
  templateUrl: './episode-detail.html',
  styleUrl: './episode-detail.scss',
})
export class EpisodeDetail implements OnInit, OnDestroy {
  private readonly uploads = inject(Uploads);
  private readonly jobsService = inject(Jobs);
  private readonly router = inject(Router);

  /** Route param, bound by withComponentInputBinding(). */
  readonly id = input.required<string>();

  private static readonly JOB_POLL_INTERVAL_MS = 3000;
  private pollSubscription?: Subscription;

  protected readonly upload = signal<UploadOut | null>(null);
  protected readonly jobs = signal<JobOut[]>([]);
  protected readonly loading = signal(true);
  protected readonly errorMessage = signal<string | null>(null);
  protected readonly pendingDelete = signal<UploadOut | null>(null);
  protected readonly name = computed(() => {
    const u = this.upload();
    return u ? displayName(u.filename) : '';
  });
  protected readonly duration = computed(() => formatDuration(this.upload()?.duration_seconds ?? null));
  protected readonly compact = compact;

  // ---- YouTube link (docs/decisions/0022 §1) ----------------------------------------
  protected readonly editingLink = signal(false);
  protected readonly linkDraft = signal('');
  protected readonly linkError = signal<string | null>(null);

  protected startEditLink(): void {
    this.linkDraft.set(this.upload()?.youtube_video_id ?? '');
    this.linkError.set(null);
    this.editingLink.set(true);
  }

  protected saveLink(): void {
    const value = this.linkDraft().trim();
    if (value && !/^[A-Za-z0-9_-]{11}$/.test(value)) {
      this.linkError.set('A YouTube video ID has 11 characters (letters, digits, - and _).');
      return;
    }
    this.uploads.linkYoutube(this.uploadId(), value || null).subscribe({
      next: (updated) => {
        this.upload.set(updated);
        this.editingLink.set(false);
      },
      error: () => this.linkError.set('Could not save the link.'),
    });
  }

  private uploadId(): number {
    return Number(this.id());
  }

  ngOnInit(): void {
    this.uploads.getUpload(this.uploadId()).subscribe({
      next: (upload) => {
        this.upload.set(upload);
        this.loading.set(false);
      },
      error: () => {
        this.errorMessage.set('Could not load this episode. It may have been deleted.');
        this.loading.set(false);
      },
    });
    this.refreshJobs();
    this.pollSubscription = interval(EpisodeDetail.JOB_POLL_INTERVAL_MS).subscribe(() => this.refreshJobs());
  }

  ngOnDestroy(): void {
    this.pollSubscription?.unsubscribe();
  }

  /** Polling tick (docs/decisions/0006). Failures are swallowed; the next tick retries. */
  protected refreshJobs(): void {
    this.jobsService.listJobsForUpload(this.uploadId()).subscribe({
      next: (jobs) => this.jobs.set(jobs),
      error: () => undefined,
    });
  }

  protected confirmDelete(): void {
    const target = this.pendingDelete();
    if (!target) return;
    this.uploads.deleteUpload(target.id).subscribe({
      next: () => {
        this.pendingDelete.set(null);
        this.router.navigateByUrl('/manage');
      },
      error: () => {
        this.pendingDelete.set(null);
        this.errorMessage.set('Could not delete this episode.');
      },
    });
  }
}
