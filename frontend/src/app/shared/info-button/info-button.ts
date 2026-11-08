import { Component, computed, inject, input } from '@angular/core';

import { Explain } from '../../explain/explain';
import { DocKey, TOOL_DOCS } from '../../explain/tool-docs';

/** ⓘ button: opens the explanation popup for a tool or platform process. */
@Component({
  selector: 'app-info',
  imports: [],
  template: `<button
    type="button"
    class="info-btn"
    [attr.aria-label]="'How ' + title() + ' works'"
    [attr.title]="'How ' + title() + ' works'"
    (click)="explain.open(key()); $event.stopPropagation()"
  >
    i
  </button>`,
})
export class InfoButton {
  protected readonly explain = inject(Explain);
  readonly key = input.required<DocKey>();
  protected readonly title = computed(() => TOOL_DOCS[this.key()].title);
}
