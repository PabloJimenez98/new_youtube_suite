import { ComponentFixture, TestBed } from '@angular/core/testing';

import { PreviewBadge } from './preview-badge';

describe('PreviewBadge', () => {
  let fixture: ComponentFixture<PreviewBadge>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({ imports: [PreviewBadge] }).compileComponents();
    fixture = TestBed.createComponent(PreviewBadge);
  });

  it('defaults to the label "Preview"', () => {
    fixture.detectChanges();
    expect((fixture.nativeElement as HTMLElement).textContent?.trim()).toBe('Preview');
  });

  it('renders a custom label', () => {
    fixture.componentRef.setInput('label', 'Coming soon');
    fixture.detectChanges();
    expect((fixture.nativeElement as HTMLElement).textContent?.trim()).toBe('Coming soon');
  });
});
