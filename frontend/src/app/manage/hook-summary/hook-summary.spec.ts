import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ClipOut } from '../../shorts/shorts';
import { HookSummary, hookRows } from './hook-summary';

const weights = { start: 0.3, pace: 0.25, framing: 0.25, address: 0.2 };
function clip(id: number, start: number, score: number | null, rank = 1, subs = { start: 1, pace: 0.9, framing: 0.3, address: 0.8 }): ClipOut {
  return {
    id,
    rank: id,
    start,
    end: start + 30,
    duration: 30,
    score: 0.8,
    aspect_ratio: '9:16',
    caption_burned_in: true,
    framing: {},
    hook:
      score === null
        ? null
        : {
            score,
            rank,
            scored_at: '2026-11-08T12:00:00Z',
            features: { sub_scores: subs, weights, features: { lead_silence_s: 0.2, words_per_second: 2.9, first_is_question: true } },
          },
  };
}

describe('hookRows', () => {
  it('ranks scored clips and names the strongest and weakest signal', () => {
    const rows = hookRows([clip(1, 60, 40, 2), clip(2, 300, 81, 1), clip(3, 900, null)]);
    expect(rows.map((r) => r.clip.id)).toEqual([2, 1]);
    expect(rows[0].strongest).toBe('Starts at once');
    expect(rows[0].weakest).toBe('Question / curiosity');
    expect(rows[0].question).toBe(true);
    expect(rows[0].subs.map((s) => s.weight)).toEqual([0.3, 0.25, 0.25, 0.2]);
  });
});

describe('HookSummary', () => {
  let fixture: ComponentFixture<HookSummary>;
  const el = () => fixture.nativeElement as HTMLElement;

  beforeEach(async () => {
    await TestBed.configureTestingModule({ imports: [HookSummary] }).compileComponents();
    fixture = TestBed.createComponent(HookSummary);
  });

  it('shows the weights, one row per scored clip and a note for unscored ones', () => {
    fixture.componentRef.setInput('clips', [clip(1, 60, 40, 2), clip(2, 300, 81, 1), clip(3, 900, null)]);
    fixture.detectChanges();
    expect(el().querySelector('.hs__lead')!.textContent).toContain('Starts at once 0.3');
    const rows = el().querySelectorAll('tbody tr');
    expect(rows.length).toBe(2);
    expect(rows[0].querySelector('.hs__score-num')!.textContent).toBe('81');
    expect(rows[0].querySelector('.hs__measured')!.textContent).toContain('0.2 s silence');
    expect(el().textContent).toContain('1 clip(s) not scored yet');
  });

  it('seeks to a clip’s start', () => {
    fixture.componentRef.setInput('clips', [clip(2, 300, 81, 1)]);
    fixture.detectChanges();
    const spy = vi.fn();
    fixture.componentInstance.seek.subscribe(spy);
    el().querySelector<HTMLButtonElement>('.hs__clip')!.click();
    expect(spy).toHaveBeenCalledWith(300);
  });

  it('invites a hook run when nothing is scored', () => {
    fixture.componentRef.setInput('clips', [clip(3, 900, null)]);
    fixture.detectChanges();
    expect(el().textContent).toContain('Run Hook score above');
  });
});
