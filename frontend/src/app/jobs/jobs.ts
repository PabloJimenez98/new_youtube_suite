import { HttpClient } from '@angular/common/http';
import { Service, inject } from '@angular/core';
import { Observable } from 'rxjs';

import { JobOut, ToolType } from './job.model';

@Service()
export class Jobs {
  private readonly http = inject(HttpClient);

  /**
   * POST /uploads/{id}/jobs -- creates a `queued` job row. Nothing
   * processes it yet (that's M2+); this just records the request.
   */
  createJob(uploadId: number, toolType: ToolType): Observable<JobOut> {
    return this.http.post<JobOut>(`/uploads/${uploadId}/jobs`, {
      tool_type: toolType,
    });
  }

  listJobsForUpload(uploadId: number): Observable<JobOut[]> {
    return this.http.get<JobOut[]>(`/uploads/${uploadId}/jobs`);
  }

  listAllJobs(): Observable<JobOut[]> {
    return this.http.get<JobOut[]>('/jobs');
  }
}
