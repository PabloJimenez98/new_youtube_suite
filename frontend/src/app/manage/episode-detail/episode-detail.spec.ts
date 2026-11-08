import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting } from '@angular/common/http/testing';
import { ComponentFixture, TestBed } from '@angular/core/testing';
import { Router, provideRouter } from '@angular/router';
import { of, throwError } from 'rxjs';

import { Jobs } from '../../jobs/jobs';
import { UploadOut } from '../../uploads/upload.model';
import { Uploads } from '../../uploads/uploads';
import { EpisodeDetail } from './episode-detail';

const upload: UploadOut = {
  id: 7,
  filename: 'SHOW 2X07： Ondas [jQ0W3HEcO7M].mkv',
  storage_path: 'x',
  duration_seconds: 4000,
  uploaded_at: '2026-08-14T12:00:00Z',
  status: 'uploaded',
};

describe('EpisodeDetail', () => {
  let fixture: ComponentFixture<EpisodeDetail>;
  let uploadsStub: { getUpload: ReturnType<typeof vi.fn>; deleteUpload: ReturnType<typeof vi.fn>; linkYoutube: ReturnType<typeof vi.fn> };
  let jobsStub: { listJobsForUpload: ReturnType<typeof vi.fn> };
  const el = () => fixture.nativeElement as HTMLElement;

  async function setup(getUpload = vi.fn().mockReturnValue(of(upload))) {
    vi.useFakeTimers();
    uploadsStub = {
      getUpload,
      deleteUpload: vi.fn().mockReturnValue(of(undefined)),
      linkYoutube: vi.fn((id: number, v: string | null) => of({ ...upload, youtube_video_id: v })),
    };
    jobsStub = { listJobsForUpload: vi.fn().mockReturnValue(of([])) };
    await TestBed.configureTestingModule({
      imports: [EpisodeDetail],
      providers: [
        { provide: Uploads, useValue: uploadsStub },
        { provide: Jobs, useValue: jobsStub },
        provideRouter([]),
        provideHttpClient(),
        provideHttpClientTesting(),
      ],
    }).compileComponents();
    fixture = TestBed.createComponent(EpisodeDetail);
    fixture.componentRef.setInput('id', '7');
    fixture.detectChanges();
  }

  afterEach(() => {
    fixture.destroy();
    vi.useRealTimers();
  });

  it('loads the episode from the route id and shows its clean name and duration', async () => {
    await setup();
    expect(uploadsStub.getUpload).toHaveBeenCalledWith(7);
    expect(el().querySelector('h1')!.textContent!.trim()).toBe('SHOW 2X07： Ondas');
    expect(el().querySelector('.detail__meta')!.textContent).toContain('1:06:40');
    expect(el().querySelector('app-episode-workspace')).not.toBeNull();
  });

  it('polls only this episode’s jobs', async () => {
    await setup();
    expect(jobsStub.listJobsForUpload).toHaveBeenCalledTimes(1);
    vi.advanceTimersByTime(3000);
    expect(jobsStub.listJobsForUpload).toHaveBeenCalledTimes(2);
    expect(jobsStub.listJobsForUpload).toHaveBeenCalledWith(7);
  });

  it('shows an error when the episode can’t be loaded', async () => {
    await setup(vi.fn().mockReturnValue(throwError(() => new Error('404'))));
    expect(el().textContent).toContain('Could not load this episode');
  });

  it('deletes after confirmation and goes back to the list', async () => {
    await setup();
    const navigate = vi.spyOn(TestBed.inject(Router), 'navigateByUrl').mockResolvedValue(true);
    el().querySelector<HTMLButtonElement>('.detail__delete')!.click();
    fixture.detectChanges();
    expect(fixture.componentInstance['pendingDelete']()).toEqual(upload);

    fixture.componentInstance['confirmDelete']();
    expect(uploadsStub.deleteUpload).toHaveBeenCalledWith(7);
    expect(navigate).toHaveBeenCalledWith('/manage');
  });

  it('shows the linked video’s YouTube numbers and a link to watch it', async () => {
    await setup(
      vi.fn().mockReturnValue(
        of({
          ...upload,
          youtube_video_id: 'jQ0W3HEcO7M',
          youtube: { title: 't', published_at: '2026-03-01T10:00:00Z', view_count: 123456, like_count: 2000, comment_count: 5, thumbnail_url: null, snapshot_date: '2026-11-08' },
        }),
      ),
    );
    const yt = el().querySelector('.detail__yt')!;
    expect(yt.textContent).toContain('123K views');
    expect(yt.querySelector('a')!.getAttribute('href')).toBe('https://www.youtube.com/watch?v=jQ0W3HEcO7M');
  });

  it('links a video by hand, validating the ID first', async () => {
    await setup();
    el().querySelector<HTMLButtonElement>('.detail__yt .link-btn')!.click();
    fixture.detectChanges();
    const box = el().querySelector<HTMLInputElement>('.detail__yt-input')!;
    box.value = 'short';
    box.dispatchEvent(new Event('input'));
    fixture.componentInstance['saveLink']();
    fixture.detectChanges();
    expect(uploadsStub.linkYoutube).not.toHaveBeenCalled();
    expect(el().textContent).toContain('11 characters');

    box.value = 'oSCkZ8yVz7M';
    box.dispatchEvent(new Event('input'));
    fixture.componentInstance['saveLink']();
    fixture.detectChanges();
    expect(uploadsStub.linkYoutube).toHaveBeenCalledWith(7, 'oSCkZ8yVz7M');
    expect(el().querySelector('.detail__yt-input')).toBeNull();
  });
});
