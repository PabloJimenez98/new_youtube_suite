import { ComponentFixture, TestBed } from '@angular/core/testing';

import { Sparkline } from './sparkline';

describe('Sparkline', () => {
  let fixture: ComponentFixture<Sparkline>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({ imports: [Sparkline] }).compileComponents();
    fixture = TestBed.createComponent(Sparkline);
  });

  it('renders one point per value', () => {
    fixture.componentRef.setInput('values', [1, 5, 3, 8]);
    fixture.detectChanges();

    const polyline = (fixture.nativeElement as HTMLElement).querySelector('polyline');
    const points = polyline?.getAttribute('points')?.trim().split(' ') ?? [];
    expect(points.length).toBe(4);
  });

  it('renders nothing for an empty series without erroring', () => {
    fixture.componentRef.setInput('values', []);
    expect(() => fixture.detectChanges()).not.toThrow();
    const polyline = (fixture.nativeElement as HTMLElement).querySelector('polyline');
    expect(polyline?.getAttribute('points')).toBe('');
  });
});
