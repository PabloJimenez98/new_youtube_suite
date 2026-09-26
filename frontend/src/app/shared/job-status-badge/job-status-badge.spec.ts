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

  it('shows the error message as a tooltip on failed jobs', () => {
    fixture.componentRef.setInput('status', 'failed');
    fixture.componentRef.setInput('error', 'No transcript for this upload -- run Transcript first');
    fixture.detectChanges();

    const el = (fixture.nativeElement as HTMLElement).querySelector('span');
    expect(el?.getAttribute('title')).toBe(
      'No transcript for this upload -- run Transcript first',
    );
    expect(el?.classList.contains('badge--has-tooltip')).toBe(true);
    expect(el?.classList.contains('badge--failed')).toBe(true);
  });

  it.each(['queued', 'running', 'succeeded'] as const)(
    'has no tooltip when %s, even with a stale error',
    (status) => {
      fixture.componentRef.setInput('status', status);
      fixture.componentRef.setInput('error', 'old error');
      fixture.detectChanges();

      const el = (fixture.nativeElement as HTMLElement).querySelector('span');
      expect(el?.hasAttribute('title')).toBe(false);
    },
  );

  it('has no tooltip for a failed job without a message', () => {
    fixture.componentRef.setInput('status', 'failed');
    fixture.detectChanges();

    const el = (fixture.nativeElement as HTMLElement).querySelector('span');
    expect(el?.hasAttribute('title')).toBe(false);
  });
});
