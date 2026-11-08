import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ClipOut, HookScore } from '../../shorts/shorts';
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
  hook: null,
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

  describe('hook score (M14)', () => {
    const hook = (score: number, rank: number, fallback = false): HookScore => ({
      score,
      rank,
      scored_at: '2026-11-08T12:00:00Z',
      features: {
        language: fallback ? 'en' : 'es',
        language_fallback: fallback,
        sub_scores: { start: 1, pace: 0.67, framing: 1, address: 0.8 },
        weights: { start: 0.3, pace: 0.25, framing: 0.25, address: 0.2 },
        features: { lead_silence_s: 0.2, words_per_second: 1.87, first_is_question: true, emotion_max: 0.99, readability: null },
      },
    });
    const badges = () => [...el().querySelectorAll('.shorts__hook-badge')].map((b) => b.textContent!.replace(/\s+/g, ' ').trim());

    it('shows "not scored" and no relative-score hint before a hook run', () => {
      expect(badges()).toEqual(['Hook: not scored', 'Hook: not scored']);
      expect(el().querySelector('.shorts__hint')).toBeNull();
      expect(el().querySelector('.shorts__why')).toBeNull();
    });

    it('shows score and rank, with the not-a-retention-% wording', () => {
      fixture.componentRef.setInput('clips', [
        { ...clip(11, 1, 2047.2, 0.78), hook: hook(72.4, 2) },
        { ...clip(12, 2, 1874.8, 0.67), hook: hook(90, 1) },
      ]);
      fixture.detectChanges();
      expect(badges()).toEqual(['Hook 72 · #2', 'Hook 90 · #1']);
      expect(el().querySelector('.shorts__hint')!.textContent).toContain('not a retention %');
    });

    it('expands the breakdown: sub-scores × weights, inputs, weight-0 features, language', () => {
      fixture.componentRef.setInput('clips', [{ ...clip(11, 1, 10, 0.78), hook: hook(80, 1, true) }]);
      fixture.detectChanges();
      expect(el().querySelector('.shorts__breakdown')).toBeNull();

      (el().querySelector('.shorts__why') as HTMLButtonElement).click();
      fixture.detectChanges();
      const subs = [...el().querySelectorAll('.shorts__subs li')].map((li) =>
        [...li.querySelectorAll('span')].map((s) => s.textContent!.trim()).join(' '),
      );
      expect(subs).toEqual([
        'Starts at once (no dead air) 1.00 × 0.3',
        'Pace (words/s) 0.67 × 0.25',
        'Opens with a question / curiosity gap 1.00 × 0.25',
        'Direct address ("you", imperatives) 0.80 × 0.2',
      ]);
      const inputs = [...el().querySelectorAll('.shorts__inputs li')].map((li) => li.textContent!.trim());
      expect(inputs).toEqual(['lead_silence_s: 0.20', 'words_per_second: 1.87', 'first_is_question: true']);
      const logged = [...el().querySelectorAll('.shorts__logged li')].map((li) => li.textContent!.trim());
      expect(logged).toEqual(['emotion_max: 0.99', 'readability: –']);
      expect(el().querySelector('.shorts__breakdown')!.textContent).toContain('language: en (fallback)');

      (el().querySelector('.shorts__why') as HTMLButtonElement).click();
      fixture.detectChanges();
      expect(el().querySelector('.shorts__breakdown')).toBeNull();
    });
  });
});
