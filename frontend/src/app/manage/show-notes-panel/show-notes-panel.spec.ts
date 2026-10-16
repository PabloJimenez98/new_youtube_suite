import { ComponentFixture, TestBed } from '@angular/core/testing';
import { of, throwError } from 'rxjs';

import { ShowNotes, ShowNotesOut } from '../../shownotes/shownotes';
import { ShowNotesPanel } from './show-notes-panel';

const notes: ShowNotesOut = {
  title_suggestions: ['First title', 'Second title', 'Third title'],
  description: 'An episode.\n\n00:00 Intro',
  tags: ['ai', 'deep learning'],
  summary: 'They talk.',
  language: 'en',
  llm_model_used: 'Qwen3.5-4B-Q4_K_M+shownotes-v1',
  created_at: '2026-10-16T12:00:00Z',
  updated_at: '2026-10-16T12:00:00Z',
};

describe('ShowNotesPanel', () => {
  let fixture: ComponentFixture<ShowNotesPanel>;
  let serviceStub: { updateShowNotes: ReturnType<typeof vi.fn> };
  const el = () => fixture.nativeElement as HTMLElement;
  const saveButton = () => el().querySelector<HTMLButtonElement>('.show-notes__save')!;

  function type(selector: string, value: string): void {
    const input = el().querySelector<HTMLInputElement | HTMLTextAreaElement>(selector)!;
    input.value = value;
    input.dispatchEvent(new Event('input'));
    fixture.detectChanges();
  }

  beforeEach(async () => {
    serviceStub = { updateShowNotes: vi.fn() };
    await TestBed.configureTestingModule({
      imports: [ShowNotesPanel],
      providers: [{ provide: ShowNotes, useValue: serviceStub }],
    }).compileComponents();
    fixture = TestBed.createComponent(ShowNotesPanel);
    fixture.componentRef.setInput('uploadId', 5);
    fixture.componentRef.setInput('notes', notes);
    fixture.detectChanges();
  });

  it('renders every field from the draft', () => {
    const titles = [...el().querySelectorAll<HTMLInputElement>('.show-notes__title-row input')].map((i) => i.value);
    expect(titles).toEqual(notes.title_suggestions);
    expect(el().querySelector<HTMLTextAreaElement>('#sn-description-5')!.value).toBe(notes.description);
    expect(el().querySelector<HTMLInputElement>('#sn-tags-5')!.value).toBe('ai, deep learning');
    expect(el().querySelector<HTMLTextAreaElement>('#sn-summary-5')!.value).toBe('They talk.');
    expect(el().textContent).toContain('Qwen3.5-4B-Q4_K_M+shownotes-v1');
  });

  it('Save is disabled until something is edited', () => {
    expect(saveButton().disabled).toBe(true);
    type('#sn-summary-5', 'Edited.');
    expect(saveButton().disabled).toBe(false);
  });

  it('Save PATCHes the edited draft and reloads from the response', () => {
    serviceStub.updateShowNotes.mockReturnValue(of({ ...notes, summary: 'Edited.', tags: ['x', 'y'] }));
    type('#sn-summary-5', 'Edited.');
    type('#sn-tags-5', ' x ,, y ');
    saveButton().click();
    fixture.detectChanges();

    expect(serviceStub.updateShowNotes).toHaveBeenCalledWith(5, {
      title_suggestions: notes.title_suggestions,
      description: notes.description,
      tags: ['x', 'y'],
      summary: 'Edited.',
    });
    expect(el().querySelector<HTMLInputElement>('#sn-tags-5')!.value).toBe('x, y');
    expect(saveButton().disabled).toBe(true); // pristine again
  });

  it('shows the backend error when Save is rejected', () => {
    serviceStub.updateShowNotes.mockReturnValue(
      throwError(() => ({ error: { detail: 'tags exceed YouTube limit' } })),
    );
    type('#sn-summary-5', 'Edited.');
    saveButton().click();
    fixture.detectChanges();
    expect(el().querySelector('[role="alert"]')?.textContent).toContain('tags exceed YouTube limit');
  });

  it('cannot save an empty description', () => {
    type('#sn-description-5', '');
    expect(saveButton().disabled).toBe(true);
  });

  it('copy buttons write the expected text', async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, 'clipboard', { value: { writeText }, configurable: true });
    const buttons = [...el().querySelectorAll<HTMLButtonElement>('.show-notes__copy')];

    buttons[1].click(); // second title
    buttons[3].click(); // description
    buttons[4].click(); // tags
    await Promise.resolve();
    fixture.detectChanges();

    expect(writeText.mock.calls.map((c) => c[0])).toEqual([
      'Second title',
      notes.description,
      'ai, deep learning',
    ]);
    expect(buttons[4].textContent?.trim()).toBe('Copied!');
  });

  it('a new draft replaces the form contents', () => {
    type('#sn-summary-5', 'Unsaved edit');
    fixture.componentRef.setInput('notes', { ...notes, summary: 'From a re-run.' });
    fixture.detectChanges();
    expect(el().querySelector<HTMLTextAreaElement>('#sn-summary-5')!.value).toBe('From a re-run.');
  });
});
