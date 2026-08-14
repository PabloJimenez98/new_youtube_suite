import { Component, OnInit, computed, inject, signal } from '@angular/core';
import { forkJoin } from 'rxjs';

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
export class Manage implements OnInit {
  private readonly uploads = inject(Uploads);
  private readonly jobsService = inject(Jobs);

  protected readonly videos = signal<UploadOut[]>([]);
  protected readonly loading = signal(true);
  protected readonly errorMessage = signal<string | null>(null);
  protected readonly pendingDelete = signal<UploadOut | null>(null);
  protected readonly deleting = signal(false);

  private readonly jobsByUploadMap = signal<Map<number, JobOut[]>>(new Map());
  protected readonly jobsFor = computed(() => this.jobsByUploadMap());

  ngOnInit(): void {
    this.refresh();
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

        const grouped = new Map<number, JobOut[]>();
        for (const job of jobs) {
          const list = grouped.get(job.upload_id) ?? [];
          list.push(job);
          grouped.set(job.upload_id, list);
        }
        this.jobsByUploadMap.set(grouped);
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
