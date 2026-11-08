import { ComponentFixture, TestBed } from '@angular/core/testing';

import { Explain } from '../../explain/explain';
import { TOOL_DOCS } from '../../explain/tool-docs';
import { ExplainDialog } from './explain-dialog';

describe('ExplainDialog', () => {
  let fixture: ComponentFixture<ExplainDialog>;
  let explain: Explain;
  const el = () => fixture.nativeElement as HTMLElement;

  beforeEach(async () => {
    await TestBed.configureTestingModule({ imports: [ExplainDialog] }).compileComponents();
    fixture = TestBed.createComponent(ExplainDialog);
    explain = TestBed.inject(Explain);
    fixture.detectChanges();
  });

  it('renders nothing while closed', () => {
    expect(el().querySelector('[role=dialog]')).toBeNull();
  });

  it('shows the doc: lead, numbered steps, flow, model card, metric and limits', () => {
    explain.open('relevance');
    fixture.detectChanges();
    const doc = TOOL_DOCS.relevance;

    expect(el().querySelector('#explain-title')!.textContent).toBe(doc.title);
    expect(el().querySelector('.explain__lead')!.textContent).toBe(doc.oneLiner);
    expect(el().querySelectorAll('.explain__steps > li').length).toBe(doc.steps.length);
    expect(el().querySelectorAll('.flow__node').length).toBe(doc.flow.nodes.length);
    expect(el().querySelector('.explain__card')!.textContent).toContain(doc.model.name);
    expect(el().textContent).toContain(doc.metric!.doesNotMean);
    expect(el().querySelectorAll('.explain__limits li').length).toBe(doc.limitations.length);
    expect(el().querySelectorAll('app-term').length).toBeGreaterThan(0);
  });

  it('switches a custom model between its inference and training flows', () => {
    explain.open('chapters');
    fixture.detectChanges();
    const [inference, training] = [...el().querySelectorAll<HTMLButtonElement>('.segmented button')];
    expect(inference.getAttribute('aria-pressed')).toBe('true');

    training.click();
    fixture.detectChanges();
    expect(el().querySelectorAll('.flow__node').length).toBe(TOOL_DOCS.chapters.training!.flow.nodes.length);
    expect(el().textContent).toContain('VidChapters-7M');
  });

  it('has no training toggle for pretrained tools, and resets the view on reopen', () => {
    explain.open('chapters');
    fixture.detectChanges();
    el().querySelectorAll<HTMLButtonElement>('.segmented button')[1].click();
    explain.open('transcribe');
    fixture.detectChanges();
    expect(el().querySelector('.segmented')).toBeNull();
    expect(fixture.componentInstance['view']()).toBe('inference');
  });

  it('closes on Escape, the close button and a backdrop click', () => {
    explain.open('hook');
    fixture.detectChanges();
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));
    expect(explain.openKey()).toBeNull();

    explain.open('hook');
    fixture.detectChanges();
    el().querySelector<HTMLButtonElement>('[aria-label=Close]')!.click();
    expect(explain.openKey()).toBeNull();

    explain.open('hook');
    fixture.detectChanges();
    el().querySelector<HTMLElement>('.explain')!.click();
    expect(explain.openKey()).toBe('hook'); // clicks inside the panel don't close it
    el().querySelector<HTMLElement>('.dialog-backdrop')!.click();
    expect(explain.openKey()).toBeNull();
  });
});
