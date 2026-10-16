import { ComponentFixture, TestBed } from '@angular/core/testing';

import { RelevanceOut } from '../../relevance/relevance';
import { RelevancePanel } from './relevance-panel';

const relevance: RelevanceOut = {
  version: 'heuristic-v0+Qwen+relevance-p1',
  windows: [
    { start: 0, end: 15, score: 0 },
    { start: 7.5, end: 22.5, score: 1 },
    { start: 15, end: 30, score: 0.5 },
  ],
  top_moments: [
    { start: 7.5, end: 22.5, score: 1, text: 'The best bit' },
    { start: 15, end: 30, score: 0.5, text: 'x'.repeat(200) },
  ],
};

describe('RelevancePanel', () => {
  let fixture: ComponentFixture<RelevancePanel>;
  const el = () => fixture.nativeElement as HTMLElement;

  beforeEach(async () => {
    await TestBed.configureTestingModule({ imports: [RelevancePanel] }).compileComponents();
    fixture = TestBed.createComponent(RelevancePanel);
    fixture.componentRef.setInput('relevance', relevance);
    fixture.detectChanges();
  });

  it('plots each window at its midpoint on a fixed 0-1 scale', () => {
    const points = el().querySelector('polyline')!.getAttribute('points')!.split(' ');
    // duration 30 s, width 600, height 80
    expect(points).toEqual(['150.0,80.0', '300.0,0.0', '450.0,40.0']);
    expect(el().querySelectorAll('circle').length).toBe(2);
  });

  it('lists the top moments with time, score and trimmed text', () => {
    const items = [...el().querySelectorAll('.relevance__moment')].map((b) =>
      [...b.querySelectorAll('span')].map((s) => s.textContent!.trim()),
    );
    expect(items[0]).toEqual(['00:07', '100', 'The best bit']);
    expect(items[1][2].endsWith('…')).toBe(true);
    expect(el().textContent).toContain('heuristic-v0+Qwen+relevance-p1');
  });

  it('emits seek with the moment start on click', () => {
    const seen: number[] = [];
    fixture.componentInstance.seek.subscribe((t) => seen.push(t));
    el().querySelectorAll<HTMLButtonElement>('.relevance__moment')[1].click();
    expect(seen).toEqual([15]);
  });

  it('labels the time axis', () => {
    const labels = [...el().querySelectorAll('.relevance__tick-label')].map((t) => t.textContent?.trim());
    expect(labels).toEqual(['00:00']);
  });
});
