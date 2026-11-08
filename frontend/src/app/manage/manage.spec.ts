import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting } from '@angular/common/http/testing';
import { ComponentFixture, TestBed } from '@angular/core/testing';
import { Router, provideRouter } from '@angular/router';
import { of } from 'rxjs';

import { JobOut } from '../jobs/job.model';
import { Jobs } from '../jobs/jobs';
import { UploadOut } from '../uploads/upload.model';
import { Uploads } from '../uploads/uploads';
import { Manage } from './manage';

const yt = (published: string, views: number) => ({
  title: 't',
  published_at: published,
  view_count: views,
  like_count: 10,
  comment_count: 1,
  thumbnail_url: `https://i.ytimg.com/vi/${views}/hqdefault.jpg`,
  snapshot_date: '2026-11-08',
});

const videoA: UploadOut = {
  id: 1,
  filename: 'SHOW 1X09： Torres [oSCkZ8yVz7M].mkv',
  storage_path: 'xa',
  duration_seconds: 3125,
  uploaded_at: '2026-08-14T12:00:00Z',
  status: 'uploaded',
  youtube_video_id: 'oSCkZ8yVz7M',
  youtube: yt('2026-03-01T10:00:00Z', 50000),
};
const videoB: UploadOut = { ...videoA, id: 2, filename: 'SHOW 2X01： Ondas.mkv', duration_seconds: 4000, youtube: yt('2026-10-01T10:00:00Z', 120000) };
const videoC: UploadOut = { ...videoA, id: 3, filename: 'b.mp4', duration_seconds: null, youtube_video_id: null, youtube: null };

function job(overrides: Partial<JobOut>): JobOut {
  return {
    id: 1,
    upload_id: 1,
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

const jobForA = job({ status: 'succeeded' });

const providersFor = (uploads: unknown, jobs: unknown) => [
  { provide: Uploads, useValue: uploads },
  { provide: Jobs, useValue: jobs },
  provideRouter([]),
  provideHttpClient(),
  provideHttpClientTesting(),
];

describe('Manage', () => {
  let fixture: ComponentFixture<Manage>;
  let uploadsStub: { listUploads: ReturnType<typeof vi.fn> };
  let jobsStub: { listAllJobs: ReturnType<typeof vi.fn> };
  const el = () => fixture.nativeElement as HTMLElement;
  const names = () => [...el().querySelectorAll('.ep-table__name a')].map((a) => a.textContent!.trim());
  const input = (selector: string, value: string, event = 'input') => {
    const box = el().querySelector<HTMLInputElement>(selector)!;
    box.value = value;
    box.dispatchEvent(new Event(event));
    fixture.detectChanges();
  };

  beforeEach(async () => {
    uploadsStub = { listUploads: vi.fn().mockReturnValue(of([videoA, videoB, videoC])) };
    jobsStub = { listAllJobs: vi.fn().mockReturnValue(of([jobForA])) };

    await TestBed.configureTestingModule({
      imports: [Manage],
      providers: providersFor(uploadsStub, jobsStub),
    }).compileComponents();

    fixture = TestBed.createComponent(Manage);
    fixture.detectChanges();
  });

  afterEach(() => fixture.destroy());

  it('renders a table with named column headers and one row per episode', () => {
    const headers = [...el().querySelectorAll('.ep-table thead th')].map((th) => th.textContent!.trim().replace(/[▲▼]/g, ''));
    expect(headers).toEqual(['Preview', 'Episode', 'Published', 'Views', 'Length', 'Pipeline', 'Next step']);
    expect(el().querySelectorAll('.ep-table__row').length).toBe(3);
    expect(fixture.componentInstance['jobsFor']().get(1)).toEqual([jobForA]);
  });

  it('sorts by publication date, newest first, with unlinked episodes last', () => {
    expect(names()).toEqual(['SHOW 2X01： Ondas', 'SHOW 1X09： Torres', 'b']);
  });

  it('shows the YouTube thumbnail, or a placeholder (never a <video> of the whole file)', () => {
    const rows = el().querySelectorAll('.ep-table__row');
    expect(rows[0].querySelector('img')!.getAttribute('src')).toContain('120000');
    expect(rows[2].querySelector('img')).toBeNull();
    expect(el().querySelectorAll('.ep-table video').length).toBe(0);
    expect(rows[2].querySelector('.thumb__none')).not.toBeNull();
  });

  it('shows the views compactly and links each name to its episode page', () => {
    const row = el().querySelectorAll('.ep-table__row')[1];
    expect(row.textContent).toContain('50K');
    expect(row.textContent).toContain('52:05');
    expect(row.querySelector('a')!.getAttribute('href')).toBe('/manage/1');
  });

  it('opens the episode when a row is clicked', () => {
    const navigate = vi.spyOn(TestBed.inject(Router), 'navigate').mockResolvedValue(true);
    el().querySelector<HTMLElement>('.ep-table__row')!.click();
    expect(navigate).toHaveBeenCalledWith(['/manage', 2]);
  });

  it('shows the status dots, progress and next step', () => {
    const row = el().querySelectorAll('.ep-table__row')[1];
    const dots = row.querySelectorAll('.status-dot');
    expect(dots.length).toBe(8);
    expect(dots[0].classList).toContain('status-dot--done');
    expect(row.querySelector('.ep-table__done')!.textContent).toBe('1/8');
    expect(row.querySelector('.ep-table__next')!.textContent).toContain('Speakers');
  });

  it('says "Working…" while a job is in flight', () => {
    jobsStub.listAllJobs.mockReturnValue(of([job({ upload_id: 3, status: 'running' })]));
    fixture.componentInstance['refreshJobs']();
    fixture.detectChanges();
    expect(el().querySelectorAll('.ep-table__row')[2].querySelector('.ep-table__next')!.textContent).toContain('Working…');
  });

  it('sorts by a column header, toggling the direction', () => {
    const viewsHeader = () => [...el().querySelectorAll<HTMLButtonElement>('.sort')].find((b) => b.textContent!.includes('Views'))!;
    viewsHeader().click();
    fixture.detectChanges();
    expect(names()).toEqual(['SHOW 2X01： Ondas', 'SHOW 1X09： Torres', 'b']);
    viewsHeader().click();
    fixture.detectChanges();
    expect(names()).toEqual(['SHOW 1X09： Torres', 'SHOW 2X01： Ondas', 'b']);
    expect(viewsHeader().closest('th')!.getAttribute('aria-sort')).toBe('ascending');
  });

  it('filters by search, publication date and views', () => {
    input('.manage__search', 'torres');
    expect(names()).toEqual(['SHOW 1X09： Torres']);
    input('.manage__search', '');

    const dates = el().querySelectorAll<HTMLInputElement>('input[type=date]');
    dates[0].value = '2026-06-01';
    dates[0].dispatchEvent(new Event('change'));
    fixture.detectChanges();
    expect(names()).toEqual(['SHOW 2X01： Ondas']); // and the unlinked one is excluded

    fixture.componentInstance['clearFilters']();
    fixture.detectChanges();
    const nums = el().querySelectorAll<HTMLInputElement>('input[type=number]');
    nums[1].value = '60000';
    nums[1].dispatchEvent(new Event('input'));
    fixture.detectChanges();
    expect(names()).toEqual(['SHOW 1X09： Torres']);
  });

  it('shows an empty-result row with a way to clear the filters', () => {
    input('.manage__search', 'nothing like this');
    expect(el().querySelector('.manage__no-match')).not.toBeNull();
    el().querySelector<HTMLButtonElement>('.manage__no-match .link-btn')!.click();
    fixture.detectChanges();
    expect(el().querySelectorAll('.ep-table__row').length).toBe(3);
  });

  it('opens the upload dialog, then closes it and refetches after an upload', () => {
    expect(el().querySelector('app-upload')).toBeNull();
    el().querySelector<HTMLButtonElement>('.manage__top .btn--primary')!.click();
    fixture.detectChanges();
    expect(el().querySelector('app-upload')).not.toBeNull();

    fixture.componentInstance['onUploaded']();
    fixture.detectChanges();
    expect(el().querySelector('app-upload')).toBeNull();
    expect(uploadsStub.listUploads).toHaveBeenCalledTimes(2);
  });
});

describe('Manage empty state', () => {
  it('invites the first upload', async () => {
    await TestBed.configureTestingModule({
      imports: [Manage],
      providers: providersFor({ listUploads: vi.fn().mockReturnValue(of([])) }, { listAllJobs: vi.fn().mockReturnValue(of([])) }),
    }).compileComponents();
    const fixture = TestBed.createComponent(Manage);
    fixture.detectChanges();
    expect((fixture.nativeElement as HTMLElement).querySelector('.manage__empty')).not.toBeNull();
    fixture.destroy();
  });
});

describe('Manage job-status polling', () => {
  let fixture: ComponentFixture<Manage>;
  let jobsStub: { listAllJobs: ReturnType<typeof vi.fn> };

  beforeEach(async () => {
    vi.useFakeTimers();
    jobsStub = { listAllJobs: vi.fn().mockReturnValue(of([jobForA])) };

    await TestBed.configureTestingModule({
      imports: [Manage],
      providers: providersFor({ listUploads: vi.fn().mockReturnValue(of([videoA])) }, jobsStub),
    }).compileComponents();

    fixture = TestBed.createComponent(Manage);
    fixture.detectChanges(); // ngOnInit: initial refresh() + starts the poll timer
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('re-fetches jobs on a fixed interval', () => {
    expect(jobsStub.listAllJobs).toHaveBeenCalledTimes(1);
    vi.advanceTimersByTime(3000);
    expect(jobsStub.listAllJobs).toHaveBeenCalledTimes(2);
    vi.advanceTimersByTime(3000);
    expect(jobsStub.listAllJobs).toHaveBeenCalledTimes(3);
  });

  it('stops polling once the component is destroyed', () => {
    fixture.destroy();
    vi.advanceTimersByTime(9000);
    expect(jobsStub.listAllJobs).toHaveBeenCalledTimes(1);
  });
});
