import { ComponentFixture, TestBed } from '@angular/core/testing';

import { StatTile } from './stat-tile';

describe('StatTile', () => {
  let fixture: ComponentFixture<StatTile>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({ imports: [StatTile] }).compileComponents();
    fixture = TestBed.createComponent(StatTile);
    fixture.componentRef.setInput('label', 'Total videos');
    fixture.componentRef.setInput('value', '12');
  });

  it('renders label and value, no preview badge by default', () => {
    fixture.detectChanges();
    const el = fixture.nativeElement as HTMLElement;
    expect(el.textContent).toContain('Total videos');
    expect(el.textContent).toContain('12');
    expect(el.querySelector('app-preview-badge')).toBeNull();
  });

  it('shows a preview badge when preview=true', () => {
    fixture.componentRef.setInput('preview', true);
    fixture.detectChanges();
    expect((fixture.nativeElement as HTMLElement).querySelector('app-preview-badge')).not.toBeNull();
  });
});
