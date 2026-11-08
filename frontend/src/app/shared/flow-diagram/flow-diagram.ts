import { Component, computed, input, output } from '@angular/core';

import { Flow, FlowNodeKind, NODE_H, NODE_W, PlacedNode, layoutFlow } from '../../explain/flow';

const KIND_LABELS: Record<FlowNodeKind, string> = {
  data: 'Data',
  process: 'Transformation',
  model: 'Model',
  decision: 'Rule / scoring',
  output: 'Stored result',
  planned: 'Planned (M15)',
};

/** A flowchart of one data process, laid out top to bottom (docs/decisions/0021). */
@Component({
  selector: 'app-flow-diagram',
  imports: [],
  templateUrl: './flow-diagram.html',
  styleUrl: './flow-diagram.scss',
})
export class FlowDiagram {
  readonly flow = input.required<Flow>();
  /** Accessible name of the diagram. */
  readonly label = input('Data flow');
  readonly showLegend = input(true);
  /** Emits a node's `link` when a linked node is clicked. */
  readonly nodeClick = output<string>();

  protected readonly layout = computed(() => layoutFlow(this.flow()));
  protected readonly nodeW = NODE_W;
  protected readonly nodeH = NODE_H;
  protected readonly kinds = computed(() => {
    const present = new Set(this.flow().nodes.map((n) => n.kind));
    return (Object.keys(KIND_LABELS) as FlowNodeKind[])
      .filter((k) => present.has(k))
      .map((k) => ({ kind: k, label: KIND_LABELS[k] }));
  });

  protected activate(node: PlacedNode): void {
    if (node.link) this.nodeClick.emit(node.link);
  }
}
