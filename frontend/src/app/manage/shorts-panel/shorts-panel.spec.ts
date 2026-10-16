import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ClipOut } from '../../shorts/shorts';
import { ShortsPanel } from './shorts-panel';

const clip = (id: number, rank: number, start: number, score: number): ClipOut => ({
  id,
  rank,
  start,
  end: start + 31.5,
  duration: 31.5,
  score,
  aspect_ratio: '9:16',
  caption_burned_in: true,
  framing: { shots: [{ start: 0, end: 31.5, mode: 'crop' }] },
});

describe('ShortsPanel', () => {
  let fixture: ComponentFixture<ShortsPanel>;
  const el = () => fixture.nativeElement as HTMLElement;

  beforeEach(async () => {
    await TestBed.configureTestingModule({ imports: [ShortsPanel] }).compileComponents();
    fixture = TestBed.createComponent(ShortsPanel);
    fixture.componentRef.setInput('clips', [clip(11, 1, 2047.2, 0.78), clip(12, 2, 1874.8, 0.67)]);
    fixture.componentRef.setInput('token', 'tok en');
    fixture.detectChanges();
  });

  it('shows a vertical player per clip with the token in the URL', () => {
    const videos = [...el().querySelectorAll('video')];
    expect(videos.map((v) => v.getAttribute('src'))).toEqual([
      '/clips/11/file?token=tok%20en',
      '/clips/12/file?token=tok%20en',
    ]);
    expect(videos[0].style.aspectRatio).toBe('9 / 16');
  });

  it('lists time range and score, and links a download', () => {
    const meta = [...el().querySelectorAll('.shorts__meta')].map((m) => m.textContent!.replace(/\s+/g, ' ').trim());
    expect(meta).toEqual(['34:07–34:38 78', '31:14–31:46 67']);
    const links = [...el().querySelectorAll('a.shorts__download')].map((a) => a.getAttribute('href'));
    expect(links[0]).toBe('/clips/11/file?token=tok%20en&download=true');
  });

  it('asks the parent to seek to the clip start', () => {
    const seeks: number[] = [];
    fixture.componentInstance.seek.subscribe((t) => seeks.push(t));
    (el().querySelectorAll('.shorts__seek')[1] as HTMLButtonElement).click();
    expect(seeks).toEqual([1874.8]);
  });
});
