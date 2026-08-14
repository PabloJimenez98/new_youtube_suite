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
  protected readonly label = computed(() => LABELS[this.status()]);
}
