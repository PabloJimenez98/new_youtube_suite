import { HttpClient } from '@angular/common/http';
import { Service, inject } from '@angular/core';
import { Observable } from 'rxjs';

import { JobOut, LlmParams, ShortsParams, ToolType } from './job.model';

@Service()
export class Jobs {
  private readonly http = inject(HttpClient);

  /**
   * POST /uploads/{id}/jobs -- creates a `queued` job row for the worker.
   * `params` carries per-run options: `shorts` (M8) and the LLM choice of chapters, show notes
   * and relevance (M16).
   */
  createJob(uploadId: number, toolType: ToolType, params?: ShortsParams | LlmParams): Observable<JobOut> {
    return this.http.post<JobOut>(`/uploads/${uploadId}/jobs`, {
      tool_type: toolType,
      ...(params ? { params } : {}),
    });
  }

  listJobsForUpload(uploadId: number): Observable<JobOut[]> {
    return this.http.get<JobOut[]>(`/uploads/${uploadId}/jobs`);
  }

  listAllJobs(): Observable<JobOut[]> {
    return this.http.get<JobOut[]>('/jobs');
  }
}
