import { Component, computed, input } from '@angular/core';

import { GLOSSARY, GlossaryKey } from '../../explain/glossary';

let nextId = 0;

/** A glossary term chip with its definition in a hover/focus tooltip. */
@Component({
  selector: 'app-term',
  imports: [],
  template: `<span class="term" tabindex="0" [attr.aria-describedby]="tipId"
    >{{ entry().term }}<span class="term__tip" role="tooltip" [id]="tipId">{{ entry().definition }}</span></span
  >`,
})
export class Term {
  readonly key = input.required<GlossaryKey>();
  protected readonly entry = computed(() => GLOSSARY[this.key()]);
  protected readonly tipId = `term-tip-${nextId++}`;
}
