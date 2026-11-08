import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideRouter } from '@angular/router';

import { TOKEN_STORAGE_KEY } from '../../auth/auth';
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
    localStorage.setItem(TOKEN_STORAGE_KEY, 'fake-token');
    await TestBed.configureTestingModule({ imports: [VideoCard], providers: [provideRouter([])] }).compileComponents();
    fixture = TestBed.createComponent(VideoCard);
    fixture.componentRef.setInput('upload', upload);
  });

  afterEach(() => localStorage.removeItem(TOKEN_STORAGE_KEY));

  it('renders the clean name and status', () => {
    fixture.detectChanges();
    const el = fixture.nativeElement as HTMLElement;
    expect(el.querySelector('.video-card__title')!.textContent!.trim()).toBe('episode-04');
    expect(el.textContent).toContain('ready');
  });

  it('links to the episode page', () => {
    fixture.detectChanges();
    expect((fixture.nativeElement as HTMLElement).querySelector('a')!.getAttribute('href')).toBe('/manage/1');
  });

  it('shows the YouTube thumbnail, never a <video> of the whole file', () => {
    const el = fixture.nativeElement as HTMLElement;
    fixture.detectChanges();
    expect(el.querySelector('img')).toBeNull(); // no thumbnail: placeholder
    expect(el.querySelector('video')).toBeNull();
    fixture.componentRef.setInput('upload', {
      ...upload,
      youtube: {
        title: 't',
        published_at: '2026-01-01T00:00:00Z',
        view_count: 1,
        like_count: 0,
        comment_count: 0,
        thumbnail_url: 'https://i.ytimg.com/vi/x/hqdefault.jpg',
        snapshot_date: '2026-11-08',
      },
    });
    fixture.detectChanges();
    expect(el.querySelector('img')!.getAttribute('src')).toBe('https://i.ytimg.com/vi/x/hqdefault.jpg');
    expect(el.querySelector('video')).toBeNull();
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
