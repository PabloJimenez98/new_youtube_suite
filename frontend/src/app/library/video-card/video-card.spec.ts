import { ComponentFixture, TestBed } from '@angular/core/testing';

import { UploadOut } from '../../uploads/upload.model';
import { VideoCard } from './video-card';

const upload: UploadOut = {
  id: 1,
  filename: 'episode-04.mp4',
  storage_path: 'x_episode-04.mp4',
  duration_seconds: 125,
  uploaded_at: '2026-08-14T12:00:00Z',
  status: 'ready',
};

describe('VideoCard', () => {
  let fixture: ComponentFixture<VideoCard>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({ imports: [VideoCard] }).compileComponents();
    fixture = TestBed.createComponent(VideoCard);
    fixture.componentRef.setInput('upload', upload);
  });

  it('renders the filename and status', () => {
    fixture.detectChanges();
    const el = fixture.nativeElement as HTMLElement;
    expect(el.textContent).toContain('episode-04.mp4');
    expect(el.textContent).toContain('ready');
  });

  it('formats duration as m:ss', () => {
    fixture.detectChanges();
    expect((fixture.nativeElement as HTMLElement).textContent).toContain('2:05');
  });

  it('omits duration when null', () => {
    fixture.componentRef.setInput('upload', { ...upload, duration_seconds: null });
    fixture.detectChanges();
    expect(
      (fixture.nativeElement as HTMLElement).querySelector('.video-card__duration'),
    ).toBeNull();
  });
});
