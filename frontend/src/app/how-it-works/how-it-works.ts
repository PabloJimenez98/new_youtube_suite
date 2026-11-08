import { Component, inject } from '@angular/core';

import { Explain } from '../explain/explain';
import { GLOSSARY, GlossaryKey } from '../explain/glossary';
import { DocKey, SYSTEM_FLOW, TOOL_DOCS } from '../explain/tool-docs';
import { FlowDiagram } from '../shared/flow-diagram/flow-diagram';
import { InfoButton } from '../shared/info-button/info-button';

interface ModelRow {
  key: DocKey;
  model: string;
  today: string;
  next: string;
  label: string;
}

/**
 * The whole platform as one data flow, the status of each model, and
 * the glossary (docs/decisions/0021). Every diagram node opens its
 * explanation.
 */
@Component({
  selector: 'app-how-it-works',
  imports: [FlowDiagram, InfoButton],
  templateUrl: './how-it-works.html',
  styleUrl: './how-it-works.scss',
})
export class HowItWorks {
  private readonly explain = inject(Explain);

  protected readonly systemFlow = SYSTEM_FLOW;
  protected readonly glossary = (Object.entries(GLOSSARY) as [GlossaryKey, (typeof GLOSSARY)[GlossaryKey]][])
    .map(([key, entry]) => ({ key, ...entry }))
    .sort((a, b) => a.term.localeCompare(b.term));

  protected readonly models: ModelRow[] = [
    { key: 'transcribe', model: 'Whisper large-v3', today: 'Pretrained', next: 'Kept as is', label: '—' },
    { key: 'diarize', model: 'pyannote community-1', today: 'Pretrained', next: 'Kept as is', label: '—' },
    {
      key: 'chapters',
      model: 'BiLSTM + Qwen3.5-9B titles',
      today: 'Trained (VidChapters-7M)',
      next: 'Retrain with more Spanish data',
      label: 'Creator chapters',
    },
    {
      key: 'relevance',
      model: 'Heuristic + LLM zero-shot',
      today: 'Heuristic v0',
      next: 'BiLSTM (M15)',
      label: 'v0 pseudo-labels',
    },
    { key: 'ctr', model: 'Feature heuristic', today: 'Heuristic v0', next: 'Learned weights (M15)', label: 'Real CTR (Reporting API)' },
    { key: 'hook', model: 'Feature heuristic', today: 'Heuristic v0', next: 'LightGBM ranker (M15)', label: 'Shorts retention' },
    { key: 'shownotes', model: 'Qwen3.5-9B', today: 'LLM zero-shot', next: 'Kept as is', label: '—' },
  ];

  protected readonly docs = TOOL_DOCS;

  protected open(key: string): void {
    if (key in TOOL_DOCS) this.explain.open(key as DocKey);
  }
}
