import { Component, afterNextRender, inject } from '@angular/core';
import { ActivatedRoute } from '@angular/router';

import { Explain } from '../explain/explain';
import { GLOSSARY, GlossaryKey } from '../explain/glossary';
import { DocKey, SYSTEM_FLOW, TOOL_DOCS } from '../explain/tool-docs';
import { FlowDiagram } from '../shared/flow-diagram/flow-diagram';
import { InfoButton } from '../shared/info-button/info-button';
import { OwnLlmSection } from './own-llm-section/own-llm-section';

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
  imports: [FlowDiagram, InfoButton, OwnLlmSection],
  templateUrl: './how-it-works.html',
  styleUrl: './how-it-works.scss',
})
export class HowItWorks {
  private readonly explain = inject(Explain);

  constructor() {
    // Links such as Settings → "How our model was built" land on a section (#own-llm).
    const fragment = inject(ActivatedRoute, { optional: true })?.snapshot.fragment;
    afterNextRender(() => {
      if (fragment) document.getElementById(fragment)?.scrollIntoView({ block: 'start' });
    });
  }

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
      model: 'Heuristic + BiLSTM',
      today: 'Trained (M15)',
      next: 'Retrain as pseudo-labels grow',
      label: 'v0 pseudo-labels (Qwen teacher)',
    },
    { key: 'ctr', model: 'Feature heuristic', today: 'Heuristic v0', next: 'Learned weights (after v1)', label: 'Real CTR (Reporting API)' },
    { key: 'hook', model: 'Feature heuristic', today: 'Heuristic v0', next: 'LightGBM ranker (after v1)', label: 'Shorts retention' },
    { key: 'shownotes', model: 'Qwen3.5-9B', today: 'LLM zero-shot', next: 'Kept as is', label: '—' },
    {
      key: 'ownllm',
      model: 'yts-own-110m-dpo (100.7M)',
      today: 'Trained from scratch (M16)',
      next: 'Optional: titles only for now',
      label: 'Web text + Qwen teacher answers',
    },
  ];

  protected readonly docs = TOOL_DOCS;

  protected open(key: string): void {
    if (key in TOOL_DOCS) this.explain.open(key as DocKey);
  }
}
