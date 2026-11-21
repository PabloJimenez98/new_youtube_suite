import { Component, OnInit, inject, signal } from '@angular/core';
import { RouterLink } from '@angular/router';

import { InfoButton } from '../shared/info-button/info-button';
import { LlmChoice, LlmSettings, llmLabel } from './llm-settings';

/**
 * Settings (M16, docs/decisions/0025 §11 and its 2026-11-15 amendment): the default LLM for the
 * chapter titles, show notes and relevance tools. Each run can still override it from its tool bar.
 */
@Component({
  selector: 'app-settings',
  imports: [InfoButton, RouterLink],
  templateUrl: './settings.html',
  styleUrl: './settings.scss',
})
export class Settings implements OnInit {
  private readonly llm = inject(LlmSettings);

  protected readonly settings = this.llm.current;
  protected readonly saving = signal(false);
  protected readonly saved = signal(false);
  protected readonly error = signal<string | null>(null);
  protected readonly choices: readonly LlmChoice[] = ['qwen', 'own'];
  protected readonly label = llmLabel;

  ngOnInit(): void {
    this.llm.load().subscribe({ error: () => this.error.set("Couldn't load the settings.") });
  }

  protected choose(choice: LlmChoice): void {
    if (choice === this.settings()?.default) return;
    this.saving.set(true);
    this.saved.set(false);
    this.error.set(null);
    this.llm.setDefault(choice).subscribe({
      next: () => {
        this.saving.set(false);
        this.saved.set(true);
      },
      error: (e) => {
        this.saving.set(false);
        this.error.set(e?.error?.detail ?? "Couldn't save the setting.");
      },
    });
  }
}
