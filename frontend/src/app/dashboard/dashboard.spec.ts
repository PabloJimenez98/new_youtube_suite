import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideRouter } from '@angular/router';
import { of, throwError } from 'rxjs';

import { JobOut } from '../jobs/job.model';
import { Jobs } from '../jobs/jobs';
import { UploadOut } from '../uploads/upload.model';
import { Uploads } from '../uploads/uploads';
import { Shorts } from '../shorts/shorts';
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
  let shortsStub: { countClips: ReturnType<typeof vi.fn> };

  beforeEach(async () => {
    uploadsStub = { listUploads: vi.fn().mockReturnValue(of(videos)) };
    jobsStub = { listAllJobs: vi.fn().mockReturnValue(of(jobs)) };
    shortsStub = { countClips: vi.fn().mockReturnValue(of(7)) };

    await TestBed.configureTestingModule({
      imports: [Dashboard],
      providers: [
        { provide: Uploads, useValue: uploadsStub },
        { provide: Jobs, useValue: jobsStub },
        { provide: Shorts, useValue: shortsStub },
        provideRouter([]),
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

  it('shows real data only: no preview (dummy) tiles', () => {
    fixture.detectChanges();
    const el = fixture.nativeElement as HTMLElement;
    expect(el.querySelectorAll('app-preview-badge').length).toBe(0);
    expect(el.querySelector('.badge--preview')).toBeNull();
  });

  it('shows pipeline coverage per tool', () => {
    fixture.detectChanges();
    const coverage = fixture.componentInstance['coverage']();
    expect(coverage).toHaveLength(8);
    expect(coverage.find((c) => c.tool === 'chapters')).toMatchObject({ done: 1, total: 2, pct: 50 });
    expect(coverage.find((c) => c.tool === 'transcribe')).toMatchObject({ done: 0, total: 2 });
    expect((fixture.nativeElement as HTMLElement).querySelectorAll('.dashboard__coverage li').length).toBe(8);
  });

  it('shows the real number of generated shorts', () => {
    fixture.detectChanges();
    const tiles = [...(fixture.nativeElement as HTMLElement).querySelectorAll('app-stat-tile')];
    const shorts = tiles.find((t) => t.textContent!.includes('Shorts generated'))!;
    expect(shorts.textContent).toContain('7');
    expect(shorts.querySelector('app-preview-badge')).toBeNull();
  });
});
