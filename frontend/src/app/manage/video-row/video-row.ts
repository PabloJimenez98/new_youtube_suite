import { Component, computed, inject, input, output, signal } from '@angular/core';

import { Auth } from '../../auth/auth';
import { JobOut, TOOL_LABELS, TOOL_TYPES, ToolType } from '../../jobs/job.model';
import { Jobs } from '../../jobs/jobs';
import { UploadOut } from '../../uploads/upload.model';
import { JobStatusBadge } from '../../shared/job-status-badge/job-status-badge';

@Component({
  selector: 'app-video-row',
  imports: [JobStatusBadge],
  templateUrl: './video-row.html',
  styleUrl: './video-row.scss',
})
export class VideoRow {
  private readonly jobsService = inject(Jobs);
  private readonly auth = inject(Auth);

  readonly upload = input.required<UploadOut>();
  /** This video's jobs, as last fetched by the parent's batched call. */
  readonly jobs = input<readonly JobOut[]>([]);
  readonly deleteRequested = output<UploadOut>();

  protected readonly toolTypes = TOOL_TYPES;
  protected readonly toolLabels = TOOL_LABELS;
  protected readonly runningToolType = signal<ToolType | null>(null);

  /**
   * <video src> requests come straight from the browser, bypassing
   * Angular's HttpClient (and therefore authInterceptor) entirely --
   * no way to attach an Authorization header. The token travels as a
   * query param instead, only for this endpoint. See
   * docs/decisions/0005-video-preview-streaming.md.
   */
  protected readonly videoUrl = computed(
    () => `/uploads/${this.upload().id}/file?token=${this.auth.token() ?? ''}`,
  );

  // Jobs created via this row's own "Run" clicks, merged with whatever
  // the parent last fetched -- the parent doesn't refetch after every
  // click, so this keeps the UI current without an extra round trip.
  private readonly optimisticJobs = signal<JobOut[]>([]);
  private readonly allJobs = computed(() => [...this.optimisticJobs(), ...this.jobs()]);

  protected latestJobFor(tool: ToolType): JobOut | undefined {
    return this.allJobs()
      .filter((job) => job.tool_type === tool)
      .sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime())[0];
  }

  /**
   * "Run" is available with no job yet, and again once the latest job
   * has reached a terminal state -- hidden only while queued/running.
   * Retry policy (backoff, auto-retry) is explicitly M9's territory;
   * this is just letting the operator ask for another run by hand.
   */
  protected canRun(tool: ToolType): boolean {
    const job = this.latestJobFor(tool);
    return !job || job.status === 'succeeded' || job.status === 'failed';
  }

  protected runTool(tool: ToolType): void {
    this.runningToolType.set(tool);
    this.jobsService.createJob(this.upload().id, tool).subscribe({
      next: (job) => {
        this.optimisticJobs.update((jobs) => [job, ...jobs]);
        this.runningToolType.set(null);
      },
      error: () => {
        this.runningToolType.set(null);
      },
    });
  }
}
