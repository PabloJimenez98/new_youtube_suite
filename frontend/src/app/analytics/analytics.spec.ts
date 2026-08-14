import { ComponentFixture, TestBed } from '@angular/core/testing';

import { Analytics } from './analytics';

describe('Analytics', () => {
  let fixture: ComponentFixture<Analytics>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({ imports: [Analytics] }).compileComponents();
    fixture = TestBed.createComponent(Analytics);
  });

  it('should create', () => {
    expect(fixture.componentInstance).toBeTruthy();
  });

  it('renders the preview banner explaining this is mock data', () => {
    fixture.detectChanges();
    const el = fixture.nativeElement as HTMLElement;
    expect(el.querySelector('app-preview-banner')).not.toBeNull();
    expect(el.textContent).toContain('market-analysis pipeline ships');
  });

  it('renders the market chart with dummy trends', () => {
    fixture.detectChanges();
    expect((fixture.nativeElement as HTMLElement).querySelector('app-market-chart')).not.toBeNull();
  });
});
