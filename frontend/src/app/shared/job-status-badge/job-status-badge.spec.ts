import { ComponentFixture, TestBed } from '@angular/core/testing';

import { JobStatusBadge } from './job-status-badge';

describe('JobStatusBadge', () => {
  let fixture: ComponentFixture<JobStatusBadge>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [JobStatusBadge],
    }).compileComponents();
    fixture = TestBed.createComponent(JobStatusBadge);
  });

  it.each([
    ['queued', 'Queued'],
    ['running', 'Running'],
    ['succeeded', 'Done'],
    ['failed', 'Failed'],
  ] as const)('renders %s as "%s" with a matching badge class', (status, label) => {
    fixture.componentRef.setInput('status', status);
    fixture.detectChanges();

    const el = (fixture.nativeElement as HTMLElement).querySelector('span');
    expect(el?.textContent).toBe(label);
    expect(el?.classList.contains(`badge--${status}`)).toBe(true);
  });
});
