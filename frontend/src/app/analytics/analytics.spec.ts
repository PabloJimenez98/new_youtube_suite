import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting } from '@angular/common/http/testing';
import { ComponentFixture, TestBed } from '@angular/core/testing';

import { Analytics } from './analytics';

describe('Analytics', () => {
  let fixture: ComponentFixture<Analytics>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [Analytics],
      providers: [provideHttpClient(), provideHttpClientTesting()],
    }).compileComponents();
    fixture = TestBed.createComponent(Analytics);
  });

  it('renders the YouTube connection card (M11)', () => {
    fixture.detectChanges();
    expect((fixture.nativeElement as HTMLElement).querySelector('app-youtube-connection')).not.toBeNull();
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
