import { Component, computed, input, output } from '@angular/core';

import { SUGGESTED_ORDER, Suggestion, ToolState, doneCount } from '../../explain/pipeline';
import { TOOL_LABELS, ToolType } from '../../jobs/job.model';
import { InfoButton } from '../../shared/info-button/info-button';

const STATUS_TEXT: Record<ToolState['status'], string> = {
  done: 'Done',
  running: 'Running',
  queued: 'Queued',
  failed: 'Failed',
  ready: 'Ready to run',
  blocked: 'Waiting',
};

/**
 * The episode's pipeline at a glance and the one step to take next
 * (docs/decisions/0021). Each node opens the tool's tab; blocked nodes
 * say what they are waiting for.
 */
@Component({
  selector: 'app-pipeline-stepper',
  imports: [InfoButton],
  templateUrl: './pipeline-stepper.html',
  styleUrl: './pipeline-stepper.scss',
})
export class PipelineStepper {
  readonly states = input.required<Record<ToolType, ToolState>>();
  readonly suggestion = input<Suggestion | null>(null);
  /** A tool whose Run click is in flight. */
  readonly starting = input<ToolType | null>(null);
  readonly selectTool = output<ToolType>();
  readonly runSuggested = output<ToolType>();

  protected readonly order = SUGGESTED_ORDER;
  protected readonly labels = TOOL_LABELS;
  protected readonly statusText = STATUS_TEXT;
  protected readonly done = computed(() => doneCount(this.states()));
  protected readonly inFlight = computed(() =>
    SUGGESTED_ORDER.filter((t) => ['running', 'queued'].includes(this.states()[t].status)),
  );

  protected nodeTitle(tool: ToolType): string {
    const s = this.states()[tool];
    const reason = s.status === 'blocked' && s.blockedReason ? ` · ${s.blockedReason}` : '';
    return `${TOOL_LABELS[tool]}: ${STATUS_TEXT[s.status]}${reason}`;
  }

  /** Button text for the suggestion card. */
  protected actionLabel(s: Suggestion): string {
    const tool = s.state.tool;
    if (s.state.blockedReason) return `Open ${TOOL_LABELS[tool]}`;
    if (tool === 'shorts') return 'Set up shorts';
    return `${s.state.status === 'failed' ? 'Retry' : 'Run'} ${TOOL_LABELS[tool]}`;
  }
}
