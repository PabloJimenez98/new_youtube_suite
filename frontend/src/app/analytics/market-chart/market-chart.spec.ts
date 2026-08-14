import { ComponentFixture, TestBed } from '@angular/core/testing';

import { MarketChart } from './market-chart';

describe('MarketChart', () => {
  let fixture: ComponentFixture<MarketChart>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({ imports: [MarketChart] }).compileComponents();
    fixture = TestBed.createComponent(MarketChart);
  });

  it('renders one row per trend, scaled relative to the max', () => {
    fixture.componentRef.setInput('trends', [
      { channel: 'A', trendScore: 50 },
      { channel: 'B', trendScore: 100 },
    ]);
    fixture.detectChanges();

    const bars = (fixture.nativeElement as HTMLElement).querySelectorAll<HTMLElement>(
      '.market-chart__bar',
    );
    expect(bars.length).toBe(2);
    expect(bars[0].style.width).toBe('50%');
    expect(bars[1].style.width).toBe('100%');
  });
});
