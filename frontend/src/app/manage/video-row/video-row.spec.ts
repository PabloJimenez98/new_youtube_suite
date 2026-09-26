import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting } from '@angular/common/http/testing';
import { ComponentFixture, TestBed } from '@angular/core/testing';
import { of, throwError } from 'rxjs';

import { TOKEN_STORAGE_KEY } from '../../auth/auth';
import { JobOut } from '../../jobs/job.model';
import { Jobs } from '../../jobs/jobs';
import { UploadOut } from '../../uploads/upload.model';
import { VideoRow } from './video-row';

const upload: UploadOut = {
  id: 5,
  filename: 'episode.mp4',
  storage_path: 'x',
  duration_seconds: null,
  uploaded_at: '2026-08-14T12:00:00Z',
  status: 'uploaded',
};

function makeJob(overrides: Partial<JobOut> = {}): JobOut {
  return {
    id: 1,
    upload_id: 5,
    tool_type: 'transcribe',
    status: 'queued',
    progress_pct: null,
    started_at: null,
    completed_at: null,
    error_message: null,
    created_at: '2026-08-14T12:00:00Z',
    ...overrides,
  };
}

describe('VideoRow', () => {
  let fixture: ComponentFixture<VideoRow>;
  let jobsStub: { createJob: ReturnType<typeof vi.fn> };

  beforeEach(async () => {
    jobsStub = { createJob: vi.fn() };
    localStorage.setItem(TOKEN_STORAGE_KEY, 'fake-token');

    await TestBed.configureTestingModule({
      imports: [VideoRow],
      providers: [
        { provide: Jobs, useValue: jobsStub },
        provideHttpClient(),
        provideHttpClientTesting(),
      ],
    }).compileComponents();

    fixture = TestBed.createComponent(VideoRow);
    fixture.componentRef.setInput('upload', upload);
  });

  afterEach(() => {
    localStorage.removeItem(TOKEN_STORAGE_KEY);
  });

  it('builds the video preview URL with the upload id and auth token', () => {
    fixture.componentRef.setInput('jobs', []);
    fixture.detectChanges();

    const video = (fixture.nativeElement as HTMLElement).querySelector('video');
    expect(video?.getAttribute('src')).toBe('/uploads/5/file?token=fake-token');
  });

  it('shows a Run button for tools with no job yet', () => {
    fixture.componentRef.setInput('jobs', []);
    fixture.detectChanges();
    const buttons = (fixture.nativeElement as HTMLElement).querySelectorAll(
      '.video-row__run',
    );
    expect(buttons.length).toBe(8); // all 8 tools, none started
  });

  it('hides Run (shows only the badge) while a job is queued/running', () => {
    fixture.componentRef.setInput('jobs', [makeJob({ tool_type: 'transcribe', status: 'queued' })]);
    fixture.detectChanges();

    const badges = (fixture.nativeElement as HTMLElement).querySelectorAll(
      'app-job-status-badge',
    );
    expect(badges.length).toBe(1);
    expect(
      (fixture.nativeElement as HTMLElement).querySelectorAll('.video-row__run').length,
    ).toBe(7);
  });

  it.each(['succeeded', 'failed'] as const)(
    'shows Run again alongside the badge once the latest job is %s',
    (status) => {
      fixture.componentRef.setInput('jobs', [makeJob({ tool_type: 'transcribe', status })]);
      fixture.detectChanges();

      const badges = (fixture.nativeElement as HTMLElement).querySelectorAll(
        'app-job-status-badge',
      );
      expect(badges.length).toBe(1);
      expect(
        (fixture.nativeElement as HTMLElement).querySelectorAll('.video-row__run').length,
      ).toBe(8); // all 8 tools can be run, including the terminal transcribe one
    },
  );

  it('runTool creates a job and optimistically shows its status', () => {
    fixture.componentRef.setInput('jobs', []);
    jobsStub.createJob.mockReturnValue(of(makeJob({ tool_type: 'chapters' })));
    fixture.detectChanges();

    fixture.componentInstance['runTool']('chapters');
    fixture.detectChanges();

    expect(jobsStub.createJob).toHaveBeenCalledWith(5, 'chapters');
    const badges = (fixture.nativeElement as HTMLElement).querySelectorAll(
      'app-job-status-badge',
    );
    expect(badges.length).toBe(1);
  });

  it('emits deleteRequested with the upload on delete click', () => {
    fixture.componentRef.setInput('jobs', []);
    fixture.detectChanges();

    const spy = vi.fn();
    fixture.componentInstance.deleteRequested.subscribe(spy);
    (fixture.nativeElement as HTMLElement)
      .querySelector<HTMLButtonElement>('.video-row__delete')!
      .click();

    expect(spy).toHaveBeenCalledWith(upload);
  });

  it('resets the running state if job creation fails', () => {
    fixture.componentRef.setInput('jobs', []);
    jobsStub.createJob.mockReturnValue(throwError(() => new Error('boom')));
    fixture.detectChanges();

    fixture.componentInstance['runTool']('hook');
    expect(fixture.componentInstance['runningToolType']()).toBeNull();
  });
});
