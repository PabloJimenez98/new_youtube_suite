import { ComponentFixture, TestBed } from '@angular/core/testing';

import { PreviewBanner } from './preview-banner';

describe('PreviewBanner', () => {
  let fixture: ComponentFixture<PreviewBanner>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({ imports: [PreviewBanner] }).compileComponents();
    fixture = TestBed.createComponent(PreviewBanner);
  });

  it('renders the given message', () => {
    fixture.componentRef.setInput('message', 'This will show real data once M10 ships.');
    fixture.detectChanges();
    expect((fixture.nativeElement as HTMLElement).textContent).toContain(
      'This will show real data once M10 ships.',
    );
  });
});
