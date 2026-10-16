import { Component, effect, inject, input, signal, untracked } from '@angular/core';
import { FormArray, FormBuilder, ReactiveFormsModule, Validators } from '@angular/forms';

import { ShowNotes, ShowNotesOut, parseTags } from '../../shownotes/shownotes';

/**
 * Review/edit view for an upload's show-notes draft (M6). Every field
 * is editable; Save PATCHes the whole draft and the backend enforces
 * YouTube's limits. See docs/decisions/0011 §6.
 */
@Component({
  selector: 'app-show-notes-panel',
  imports: [ReactiveFormsModule],
  templateUrl: './show-notes-panel.html',
  styleUrl: './show-notes-panel.scss',
})
export class ShowNotesPanel {
  private readonly fb = inject(FormBuilder);
  private readonly service = inject(ShowNotes);

  readonly uploadId = input.required<number>();
  readonly notes = input.required<ShowNotesOut>();

  protected readonly saving = signal(false);
  protected readonly saveError = signal<string | null>(null);
  protected readonly copiedKey = signal<string | null>(null);
  protected readonly modelUsed = signal('');

  protected readonly form = this.fb.nonNullable.group({
    titles: this.fb.nonNullable.array<string>([]),
    description: ['', [Validators.required, Validators.maxLength(5000)]],
    tags: [''],
    summary: ['', [Validators.required, Validators.maxLength(600)]],
  });

  constructor() {
    // A new draft (first load or a re-run) replaces whatever is in the form.
    effect(() => {
      const notes = this.notes();
      untracked(() => this.load(notes));
    });
  }

  protected get titles(): FormArray {
    return this.form.controls.titles;
  }

  private load(notes: ShowNotesOut): void {
    this.titles.clear();
    for (const t of notes.title_suggestions) {
      this.titles.push(this.fb.nonNullable.control(t, [Validators.required, Validators.maxLength(100)]));
    }
    this.form.patchValue({
      description: notes.description,
      tags: notes.tags.join(', '),
      summary: notes.summary,
    });
    this.form.markAsPristine();
    this.modelUsed.set(notes.llm_model_used);
  }

  protected save(): void {
    if (this.form.invalid || this.saving()) return;
    const v = this.form.getRawValue();
    this.saving.set(true);
    this.saveError.set(null);
    this.service
      .updateShowNotes(this.uploadId(), {
        title_suggestions: v.titles.map((t) => t.trim()),
        description: v.description,
        tags: parseTags(v.tags),
        summary: v.summary,
      })
      .subscribe({
        next: (saved) => {
          this.load(saved);
          this.saving.set(false);
        },
        error: (err) => {
          const detail = err?.error?.detail;
          this.saveError.set(typeof detail === 'string' ? detail : 'Could not save -- check the limits');
          this.saving.set(false);
        },
      });
  }

  protected copy(key: string, text: string): void {
    navigator.clipboard.writeText(text).then(() => {
      this.copiedKey.set(key);
      setTimeout(() => this.copiedKey.set(null), 2000);
    });
  }

  protected copyTags(): void {
    this.copy('tags', parseTags(this.form.controls.tags.value).join(', '));
  }
}
