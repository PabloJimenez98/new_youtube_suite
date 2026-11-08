import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting } from '@angular/common/http/testing';
import { ComponentFixture, TestBed } from '@angular/core/testing';
import { of, throwError } from 'rxjs';

import { Insights } from '../insights/insights';
import { Analytics } from './analytics';
import { CHANNEL_DATA, CHANNEL_EMPTY, CLUSTERS, DRIVERS, GROWTH, KEYWORDS, OVERVIEW } from './analytics-fixtures';

describe('Analytics', () => {
  let fixture: ComponentFixture<Analytics>;
  let stub: Record<string, ReturnType<typeof vi.fn>>;
  const el = () => fixture.nativeElement as HTMLElement;
  const card = (title: string) =>
    [...el().querySelectorAll('app-chart-card')].find((c) => c.querySelector('h3')?.textContent?.trim() === title)!;

  async function setup(overrides: Partial<Record<string, ReturnType<typeof vi.fn>>> = {}) {
    stub = {
      channel: vi.fn().mockReturnValue(of(CHANNEL_EMPTY)),
      overview: vi.fn().mockReturnValue(of(OVERVIEW)),
      clusters: vi.fn().mockReturnValue(of(CLUSTERS)),
      drivers: vi.fn().mockReturnValue(of(DRIVERS)),
      growth: vi.fn().mockReturnValue(of(GROWTH)),
      keywords: vi.fn().mockReturnValue(of(KEYWORDS)),
      ...overrides,
    };
    await TestBed.configureTestingModule({
      imports: [Analytics],
      providers: [{ provide: Insights, useValue: stub }, provideHttpClient(), provideHttpClientTesting()],
    })
      .compileComponents();
    fixture = TestBed.createComponent(Analytics);
    fixture.detectChanges();
  }

  it('loads the channel and every market analysis for full episodes', async () => {
    await setup();
    expect(stub['channel']).toHaveBeenCalled();
    for (const k of ['overview', 'clusters', 'drivers', 'growth', 'keywords']) {
      expect(stub[k]).toHaveBeenCalledWith('long');
    }
  });

  it('every chart card carries an explanation tooltip', async () => {
    await setup();
    const cards = el().querySelectorAll('app-chart-card');
    expect(cards.length).toBeGreaterThanOrEqual(11);
    for (const c of cards) {
      const tip = c.querySelector('.chart-help__tip')!;
      expect(tip.textContent).toContain('What it shows');
      expect(tip.textContent).toContain('How to read it');
    }
  });

  it('shows the headline tiles for the owner’s channel', async () => {
    await setup();
    const tiles = el().querySelector('.analytics__tiles')!.textContent!;
    expect(tiles).toContain('50K'); // median views
    expect(tiles).toContain('#2 of 3'); // by median views
    expect(tiles).toContain('Video length'); // strongest |ρ|
  });

  it('switching the format refetches the market analyses', async () => {
    await setup();
    [...el().querySelectorAll<HTMLButtonElement>('.analytics__filters button')].find((b) => b.textContent!.includes('Shorts'))!.click();
    fixture.detectChanges();
    expect(stub['overview']).toHaveBeenLastCalledWith('short');
    expect(stub['keywords']).toHaveBeenLastCalledWith('short');
    expect(stub['channel']).toHaveBeenCalledTimes(1);
  });

  it('explains empty channel charts and the missing velocity data', async () => {
    await setup();
    expect(card('Daily views').textContent).toContain('Connect your channel');
    expect(card('View velocity').textContent).toContain('two daily snapshots');
  });

  it('draws the channel charts once there is data', async () => {
    await setup({ channel: vi.fn().mockReturnValue(of(CHANNEL_DATA)) });
    expect(card('Daily views').querySelector('app-chart')).not.toBeNull();
    expect(card('Real CTR').querySelector('app-chart')).not.toBeNull();
    expect(card('Audience retention').querySelector('app-chart')).not.toBeNull();
  });

  it('offers a table view of each chart', async () => {
    await setup();
    const c = card('Views per video, by channel');
    c.querySelector<HTMLButtonElement>('.chart-card__toggle')!.click();
    fixture.detectChanges();
    const rows = c.querySelectorAll('tbody tr');
    expect(rows.length).toBe(3);
    expect(rows[0].textContent).toContain('Big'); // highest median first
  });

  it('reports a failed market load', async () => {
    await setup({ overview: vi.fn().mockReturnValue(throwError(() => new Error('500'))) });
    expect(el().textContent).toContain('Could not load the market data');
  });

  it('says when the results were precomputed', async () => {
    await setup({ overview: vi.fn().mockReturnValue(of({ ...OVERVIEW, computed_at: '2026-11-08T22:21:00Z', pending: false })) });
    expect(el().querySelector('.analytics__status')!.textContent).toContain('Precomputed');
  });

  it('while the worker computes a result, says so and polls until it is stored', async () => {
    vi.useFakeTimers();
    try {
      const pending = { ...OVERVIEW, channels: [], monthly: [], pending: true, computed_at: null };
      await setup({ overview: vi.fn().mockReturnValueOnce(of(pending)).mockReturnValue(of(OVERVIEW)) });
      expect(el().querySelector('.analytics__pending')).not.toBeNull();
      expect(stub['overview']).toHaveBeenCalledTimes(1);

      vi.advanceTimersByTime(Analytics.PENDING_RETRY_MS);
      fixture.detectChanges();
      expect(stub['overview']).toHaveBeenCalledTimes(2);
      expect(el().querySelector('.analytics__pending')).toBeNull();
      expect(el().querySelector('.analytics__tiles')!.textContent).toContain('50K');

      vi.advanceTimersByTime(Analytics.PENDING_RETRY_MS * 3);
      expect(stub['overview']).toHaveBeenCalledTimes(2); // stored: no more polling
    } finally {
      vi.useRealTimers();
    }
  });
});
