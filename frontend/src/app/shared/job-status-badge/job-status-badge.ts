import { Component, computed, input } from '@angular/core';

import { JobStatus } from '../../jobs/job.model';

const LABELS: Record<JobStatus, string> = {
  queued: 'Queued',
  running: 'Running',
  succeeded: 'Done',
  failed: 'Failed',
};

@Component({
  selector: 'app-job-status-badge',
  imports: [],
  templateUrl: './job-status-badge.html',
  styleUrl: './job-status-badge.scss',
})
export class JobStatusBadge {
  readonly status = input.required<JobStatus>();
  /** The job's `error_message`, shown as a tooltip when it failed. */
  readonly error = input<string | null>(null);
  protected readonly label = computed(() => LABELS[this.status()]);
  protected readonly tooltip = computed(() =>
    this.status() === 'failed' ? this.error() : null,
  );
}
