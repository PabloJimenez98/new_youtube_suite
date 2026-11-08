import { Component, ElementRef, HostListener, computed, effect, inject, signal, viewChild } from '@angular/core';

import { TOOL_LABELS } from '../../jobs/job.model';
import { Explain } from '../../explain/explain';
import { TOOL_DOCS } from '../../explain/tool-docs';
import { FlowDiagram } from '../flow-diagram/flow-diagram';
import { Term } from '../term/term';

/**
 * The explanation popup (docs/decisions/0021): what a tool does, its
 * data flow (inference, and training for the custom models), the
 * step-by-step data process, its model card, how to read its score and
 * its limitations. One instance lives in the shell.
 */
@Component({
  selector: 'app-explain-dialog',
  imports: [FlowDiagram, Term],
  templateUrl: './explain-dialog.html',
  styleUrl: './explain-dialog.scss',
})
export class ExplainDialog {
  protected readonly explain = inject(Explain);
  private readonly panel = viewChild<ElementRef<HTMLElement>>('panel');

  protected readonly doc = computed(() => {
    const key = this.explain.openKey();
    return key ? TOOL_DOCS[key] : null;
  });
  protected readonly view = signal<'inference' | 'training'>('inference');
  protected readonly flow = computed(() => {
    const doc = this.doc();
    if (!doc) return null;
    return this.view() === 'training' && doc.training ? doc.training.flow : doc.flow;
  });
  protected readonly toolLabels = TOOL_LABELS;

  constructor() {
    // Each newly opened doc starts on its inference flow, with focus in the dialog.
    effect(() => {
      if (this.doc()) {
        this.view.set('inference');
        queueMicrotask(() => this.panel()?.nativeElement.focus());
      }
    });
  }

  @HostListener('document:keydown.escape')
  protected onEscape(): void {
    if (this.doc()) this.explain.close();
  }

  /** Keep Tab inside the dialog while it is open. */
  protected trapFocus(e: Event): void {
    const event = e as KeyboardEvent;
    const root = this.panel()?.nativeElement;
    if (!root) return;
    const focusable = [
      ...root.querySelectorAll<HTMLElement>('button, [href], [tabindex]:not([tabindex="-1"])'),
    ];
    if (!focusable.length) return;
    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    if (event.shiftKey && document.activeElement === first) {
      last.focus();
      event.preventDefault();
    } else if (!event.shiftKey && document.activeElement === last) {
      first.focus();
      event.preventDefault();
    }
  }
}
