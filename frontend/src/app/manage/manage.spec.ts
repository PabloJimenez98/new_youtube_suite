import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting } from '@angular/common/http/testing';
import { ComponentFixture, TestBed } from '@angular/core/testing';
import { of } from 'rxjs';

import { JobOut } from '../jobs/job.model';
import { Jobs } from '../jobs/jobs';
import { UploadOut } from '../uploads/upload.model';
import { Uploads } from '../uploads/uploads';
import { Manage } from './manage';

const videoA: UploadOut = {
  id: 1,
  filename: 'a.mp4',
  storage_path: 'xa',
  duration_seconds: null,
  uploaded_at: '2026-08-14T12:00:00Z',
  status: 'uploaded',
};
const videoB: UploadOut = { ...videoA, id: 2, filename: 'b.mp4' };

const jobForA: JobOut = {
  id: 1,
  upload_id: 1,
  tool_type: 'transcribe',
  status: 'queued',
  progress_pct: null,
  started_at: null,
  completed_at: null,
  error_message: null,
  created_at: '2026-08-14T12:00:00Z',
};

describe('Manage', () => {
  let fixture: ComponentFixture<Manage>;
  let uploadsStub: {
    listUploads: ReturnType<typeof vi.fn>;
    deleteUpload: ReturnType<typeof vi.fn>;
  };
  let jobsStub: { listAllJobs: ReturnType<typeof vi.fn> };

  beforeEach(async () => {
    uploadsStub = {
      listUploads: vi.fn().mockReturnValue(of([videoA, videoB])),
      deleteUpload: vi.fn().mockReturnValue(of(undefined)),
    };
    jobsStub = { listAllJobs: vi.fn().mockReturnValue(of([jobForA])) };

    await TestBed.configureTestingModule({
      imports: [Manage],
      providers: [
        { provide: Uploads, useValue: uploadsStub },
        { provide: Jobs, useValue: jobsStub },
        provideHttpClient(),
        provideHttpClientTesting(),
      ],
    }).compileComponents();

    fixture = TestBed.createComponent(Manage);
    fixture.detectChanges();
  });

  it('renders a video-row per upload, jobs grouped by upload_id', () => {
    const rows = (fixture.nativeElement as HTMLElement).querySelectorAll('app-video-row');
    expect(rows.length).toBe(2);
    expect(fixture.componentInstance['jobsFor']().get(1)).toEqual([jobForA]);
    expect(fixture.componentInstance['jobsFor']().get(2)).toBeUndefined();
  });

  it('refetches on onUploaded()', () => {
    fixture.componentInstance['onUploaded']();
    expect(uploadsStub.listUploads).toHaveBeenCalledTimes(2);
  });

  it('delete flow: request -> confirm -> removes the row and calls deleteUpload', () => {
    fixture.componentInstance['requestDelete'](videoA);
    fixture.detectChanges();
    expect(fixture.componentInstance['pendingDelete']()).toEqual(videoA);

    fixture.componentInstance['confirmDelete']();

    expect(uploadsStub.deleteUpload).toHaveBeenCalledWith(1);
    expect(fixture.componentInstance['videos']().map((v) => v.id)).toEqual([2]);
    expect(fixture.componentInstance['pendingDelete']()).toBeNull();
  });

  it('cancelDelete clears the pending video without deleting', () => {
    fixture.componentInstance['requestDelete'](videoA);
    fixture.componentInstance['cancelDelete']();

    expect(fixture.componentInstance['pendingDelete']()).toBeNull();
    expect(uploadsStub.deleteUpload).not.toHaveBeenCalled();
  });
});
