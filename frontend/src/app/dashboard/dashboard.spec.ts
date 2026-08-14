import { ComponentFixture, TestBed } from '@angular/core/testing';
import { of, throwError } from 'rxjs';

import { JobOut } from '../jobs/job.model';
import { Jobs } from '../jobs/jobs';
import { UploadOut } from '../uploads/upload.model';
import { Uploads } from '../uploads/uploads';
import { Dashboard } from './dashboard';

const videos: UploadOut[] = [
  {
    id: 1,
    filename: 'ep1.mp4',
    storage_path: 'x1',
    duration_seconds: 90,
    uploaded_at: '2026-08-14T10:00:00Z',
    status: 'uploaded',
  },
  {
    id: 2,
    filename: 'ep2.mp4',
    storage_path: 'x2',
    duration_seconds: null,
    uploaded_at: '2026-08-14T11:00:00Z',
    status: 'uploaded',
  },
];

const jobs: JobOut[] = [
  {
    id: 1,
    upload_id: 1,
    tool_type: 'transcribe',
    status: 'queued',
    progress_pct: null,
    started_at: null,
    completed_at: null,
    error_message: null,
    created_at: '2026-08-14T10:05:00Z',
  },
  {
    id: 2,
    upload_id: 1,
    tool_type: 'chapters',
    status: 'succeeded',
    progress_pct: null,
    started_at: null,
    completed_at: null,
    error_message: null,
    created_at: '2026-08-14T10:06:00Z',
  },
];

describe('Dashboard', () => {
  let fixture: ComponentFixture<Dashboard>;
  let uploadsStub: { listUploads: ReturnType<typeof vi.fn> };
  let jobsStub: { listAllJobs: ReturnType<typeof vi.fn> };

  beforeEach(async () => {
    uploadsStub = { listUploads: vi.fn().mockReturnValue(of(videos)) };
    jobsStub = { listAllJobs: vi.fn().mockReturnValue(of(jobs)) };

    await TestBed.configureTestingModule({
      imports: [Dashboard],
      providers: [
        { provide: Uploads, useValue: uploadsStub },
        { provide: Jobs, useValue: jobsStub },
      ],
    }).compileComponents();

    fixture = TestBed.createComponent(Dashboard);
  });

  it('computes total videos and runtime, noting missing durations', () => {
    fixture.detectChanges();
    const component = fixture.componentInstance;
    expect(component['totalVideos']()).toBe(2);
    expect(component['totalRuntimeLabel']().value).toBe('0h 2m'); // 90s -> 1.5m rounds to 2m
    expect(component['totalRuntimeLabel']().hint).toContain('1 video(s)');
  });

  it('breaks jobs down by status', () => {
    fixture.detectChanges();
    const breakdown = fixture.componentInstance['jobsByStatus']();
    expect(breakdown).toEqual([
      { status: 'queued', count: 1 },
      { status: 'running', count: 0 },
      { status: 'succeeded', count: 1 },
      { status: 'failed', count: 0 },
    ]);
  });

  it('shows an error message when loading fails', () => {
    uploadsStub.listUploads.mockReturnValue(throwError(() => new Error('boom')));
    fixture.detectChanges();
    expect((fixture.nativeElement as HTMLElement).textContent).toContain(
      'Could not load dashboard data.',
    );
  });

  it('renders preview badges for the dummy tiles', () => {
    fixture.detectChanges();
    const previewBadges = (fixture.nativeElement as HTMLElement).querySelectorAll(
      'app-preview-badge',
    );
    // 3 dummy stat-tiles (CTR, top hook clip, shorts generated)
    expect(previewBadges.length).toBeGreaterThanOrEqual(3);
  });
});
