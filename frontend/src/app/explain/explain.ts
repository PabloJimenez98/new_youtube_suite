import { Service, signal } from '@angular/core';

import { DocKey } from './tool-docs';

/**
 * Which explanation popup is open, app-wide. Any component can open one
 * (`<app-info>`, a How it works node, the stepper); the single
 * `<app-explain-dialog>` in the shell renders it.
 */
@Service()
export class Explain {
  readonly openKey = signal<DocKey | null>(null);

  open(key: DocKey): void {
    this.openKey.set(key);
  }

  close(): void {
    this.openKey.set(null);
  }
}
