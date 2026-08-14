import { provideHttpClient } from '@angular/common/http';
import {
  HttpTestingController,
  provideHttpClientTesting,
} from '@angular/common/http/testing';
import { TestBed } from '@angular/core/testing';

import { Jobs } from './jobs';

describe('Jobs', () => {
  let service: Jobs;
  let httpMock: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [provideHttpClient(), provideHttpClientTesting()],
    });
    service = TestBed.inject(Jobs);
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    httpMock.verify();
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });

  it('createJob POSTs { tool_type } to /uploads/{id}/jobs', () => {
    service.createJob(5, 'transcribe').subscribe();

    const req = httpMock.expectOne('/uploads/5/jobs');
    expect(req.request.method).toBe('POST');
    expect(req.request.body).toEqual({ tool_type: 'transcribe' });

    req.flush({
      id: 1,
      upload_id: 5,
      tool_type: 'transcribe',
      status: 'queued',
      progress_pct: null,
      started_at: null,
      completed_at: null,
      error_message: null,
      created_at: '2026-08-14T12:00:00Z',
    });
  });

  it('listJobsForUpload GETs /uploads/{id}/jobs', () => {
    service.listJobsForUpload(5).subscribe();

    const req = httpMock.expectOne('/uploads/5/jobs');
    expect(req.request.method).toBe('GET');
    req.flush([]);
  });

  it('listAllJobs GETs /jobs', () => {
    service.listAllJobs().subscribe();

    const req = httpMock.expectOne('/jobs');
    expect(req.request.method).toBe('GET');
    req.flush([]);
  });
});
