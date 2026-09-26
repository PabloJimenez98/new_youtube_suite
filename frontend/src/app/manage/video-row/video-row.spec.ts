import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting } from '@angular/common/http/testing';
import { ComponentFixture, TestBed } from '@angular/core/testing';
import { of, throwError } from 'rxjs';

import { TOKEN_STORAGE_KEY } from '../../auth/auth';
import { JobOut } from '../../jobs/job.model';
import { Jobs } from '../../jobs/jobs';
import { Transcripts } from '../../transcripts/transcripts';
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
  let transcriptsStub: { getSubtitle: ReturnType<typeof vi.fn> };

  beforeEach(async () => {
    jobsStub = { createJob: vi.fn() };
    transcriptsStub = { getSubtitle: vi.fn() };
    localStorage.setItem(TOKEN_STORAGE_KEY, 'fake-token');

    await TestBed.configureTestingModule({
      imports: [VideoRow],
      providers: [
        { provide: Jobs, useValue: jobsStub },
        { provide: Transcripts, useValue: transcriptsStub },
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

  describe('subtitle downloads', () => {
    const downloadButtons = () =>
      (fixture.nativeElement as HTMLElement).querySelectorAll<HTMLButtonElement>(
        '.video-row__download',
      );

    it.each(['queued', 'running', 'failed'] as const)(
      'are hidden when the only transcribe job is %s',
      (status) => {
        fixture.componentRef.setInput('jobs', [makeJob({ status })]);
        fixture.detectChanges();
        expect(downloadButtons().length).toBe(0);
      },
    );

    it('are hidden when only another tool has succeeded', () => {
      fixture.componentRef.setInput('jobs', [
        makeJob({ tool_type: 'chapters', status: 'succeeded' }),
      ]);
      fixture.detectChanges();
      expect(downloadButtons().length).toBe(0);
    });

    it('show .srt and .vtt once a transcribe job has succeeded', () => {
      fixture.componentRef.setInput('jobs', [makeJob({ status: 'succeeded' })]);
      fixture.detectChanges();
      expect([...downloadButtons()].map((b) => b.textContent?.trim())).toEqual([
        '.srt',
        '.vtt',
      ]);
    });

    it('stay available while a re-run is in flight', () => {
      fixture.componentRef.setInput('jobs', [
        makeJob({ id: 2, status: 'running', created_at: '2026-08-15T12:00:00Z' }),
        makeJob({ id: 1, status: 'succeeded' }),
      ]);
      fixture.detectChanges();
      expect(downloadButtons().length).toBe(2);
    });

    it('fetch the subtitle and save it under the video file stem', () => {
      const blob = new Blob(['1\n']);
      transcriptsStub.getSubtitle.mockReturnValue(of(blob));
      const createUrl = vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:x');
      const revokeUrl = vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => {});
      const click = vi
        .spyOn(HTMLAnchorElement.prototype, 'click')
        .mockImplementation(function (this: HTMLAnchorElement) {
          expect(this.download).toBe('episode.srt');
          expect(this.href).toBe('blob:x');
        });

      fixture.componentRef.setInput('jobs', [makeJob({ status: 'succeeded' })]);
      fixture.detectChanges();
      downloadButtons()[0].click();

      expect(transcriptsStub.getSubtitle).toHaveBeenCalledWith(5, 'srt');
      expect(createUrl).toHaveBeenCalledWith(blob);
      expect(click).toHaveBeenCalledTimes(1);
      expect(revokeUrl).toHaveBeenCalledWith('blob:x');
      expect(fixture.componentInstance['downloadingFormat']()).toBeNull();

      createUrl.mockRestore();
      revokeUrl.mockRestore();
      click.mockRestore();
    });

    it('resets the downloading state if the request fails', () => {
      transcriptsStub.getSubtitle.mockReturnValue(throwError(() => new Error('409')));
      fixture.componentRef.setInput('jobs', [makeJob({ status: 'succeeded' })]);
      fixture.detectChanges();

      fixture.componentInstance['downloadSubtitle']('vtt');
      expect(fixture.componentInstance['downloadingFormat']()).toBeNull();
    });
  });

  describe('Speakers (diarize) run button', () => {
    const speakersButton = () => {
      const cells = [
        ...(fixture.nativeElement as HTMLElement).querySelectorAll('.video-row__tool'),
      ];
      const cell = cells.find(
        (c) => c.querySelector('.video-row__tool-label')?.textContent?.trim() === 'Speakers',
      );
      return cell?.querySelector<HTMLButtonElement>('.video-row__run') ?? null;
    };

    it('is disabled with a hint until a transcript exists', () => {
      fixture.componentRef.setInput('jobs', []);
      fixture.detectChanges();

      expect(speakersButton()?.disabled).toBe(true);
      expect(speakersButton()?.getAttribute('title')).toBe('Needs a transcript');
    });

    it('stays disabled while transcription is still running', () => {
      fixture.componentRef.setInput('jobs', [makeJob({ status: 'running' })]);
      fixture.detectChanges();

      expect(speakersButton()?.disabled).toBe(true);
    });

    it('is enabled once a transcribe job has succeeded', () => {
      fixture.componentRef.setInput('jobs', [makeJob({ status: 'succeeded' })]);
      fixture.detectChanges();

      expect(speakersButton()?.disabled).toBe(false);
      expect(speakersButton()?.hasAttribute('title')).toBe(false);
    });

    it('does not gate the other tools', () => {
      fixture.componentRef.setInput('jobs', []);
      fixture.detectChanges();

      const disabled = (fixture.nativeElement as HTMLElement).querySelectorAll(
        '.video-row__run:disabled',
      );
      expect(disabled.length).toBe(1);
    });
  });

  it('passes the job error message to its status badge', () => {
    fixture.componentRef.setInput('jobs', [
      makeJob({ tool_type: 'diarize', status: 'failed', error_message: 'Diarization unavailable' }),
      makeJob({ id: 2, status: 'succeeded' }),
    ]);
    fixture.detectChanges();

    const failedBadge = (fixture.nativeElement as HTMLElement).querySelector(
      '.badge--failed',
    );
    expect(failedBadge?.getAttribute('title')).toBe('Diarization unavailable');
  });
});

