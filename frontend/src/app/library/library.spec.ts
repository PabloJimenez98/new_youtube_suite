import { ComponentFixture, TestBed } from '@angular/core/testing';
import { of, throwError } from 'rxjs';

import { UploadOut } from '../uploads/upload.model';
import { Uploads } from '../uploads/uploads';
import { Library } from './library';

const sample: UploadOut = {
  id: 1,
  filename: 'episode.mp4',
  storage_path: 'x_episode.mp4',
  duration_seconds: null,
  uploaded_at: '2026-08-14T12:00:00Z',
  status: 'uploaded',
};

describe('Library', () => {
  let fixture: ComponentFixture<Library>;
  let uploadsStub: { listUploads: ReturnType<typeof vi.fn> };

  beforeEach(async () => {
    uploadsStub = { listUploads: vi.fn().mockReturnValue(of([])) };

    await TestBed.configureTestingModule({
      imports: [Library],
      providers: [{ provide: Uploads, useValue: uploadsStub }],
    }).compileComponents();

    fixture = TestBed.createComponent(Library);
  });

  it('shows an empty state with no videos', () => {
    fixture.detectChanges();
    expect((fixture.nativeElement as HTMLElement).textContent).toContain(
      'No videos uploaded yet',
    );
  });

  it('renders a video-card per upload', () => {
    uploadsStub.listUploads.mockReturnValue(of([sample]));
    fixture.detectChanges();
    expect(
      (fixture.nativeElement as HTMLElement).querySelectorAll('app-video-card').length,
    ).toBe(1);
  });

  it('shows an error message on failure', () => {
    uploadsStub.listUploads.mockReturnValue(throwError(() => new Error('boom')));
    fixture.detectChanges();
    expect((fixture.nativeElement as HTMLElement).textContent).toContain(
      'Could not load your videos.',
    );
  });
});
