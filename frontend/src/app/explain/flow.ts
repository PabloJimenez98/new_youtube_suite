/**
 * Flowchart model + a pure layered layout for `FlowDiagram`
 * (docs/decisions/0021). Top-to-bottom: each node's layer is the
 * longest path from a source, so branches sit side by side and merges
 * land below every input.
 */
export type FlowNodeKind = 'data' | 'process' | 'model' | 'decision' | 'output' | 'planned';

export interface FlowNode {
  id: string;
  label: string;
  /** Optional second line: a parameter, a size, a model name. */
  detail?: string;
  kind: FlowNodeKind;
  /** A doc key: clicking the node opens that explanation (How it works page). */
  link?: string;
  /** Layout hint: place the node no higher than this layer (e.g. late data sources). */
  minLayer?: number;
}

export interface FlowEdge {
  from: string;
  to: string;
  label?: string;
}

export interface Flow {
  nodes: FlowNode[];
  edges: FlowEdge[];
}

export interface PlacedNode extends FlowNode {
  layer: number;
  x: number;
  y: number;
}

export interface PlacedEdge extends FlowEdge {
  path: string;
  labelX: number;
  labelY: number;
  /** Skips at least one layer and is routed through a side lane. */
  long?: boolean;
}

export interface FlowLayout {
  width: number;
  height: number;
  nodes: PlacedNode[];
  edges: PlacedEdge[];
}

export const NODE_W = 168;
export const NODE_H = 56;
const GAP_X = 28;
const GAP_Y = 44;
const PAD = 8;
const LANE = 28;

/** Throws on an edge to an unknown node or a cycle; returns the layer of each node. */
export function assignLayers(flow: Flow): Map<string, number> {
  const ids = new Set(flow.nodes.map((n) => n.id));
  const incoming = new Map<string, string[]>(flow.nodes.map((n) => [n.id, []]));
  const minLayer = new Map(flow.nodes.map((n) => [n.id, n.minLayer ?? 0]));
  for (const e of flow.edges) {
    if (!ids.has(e.from) || !ids.has(e.to)) {
      throw new Error(`flow edge ${e.from} → ${e.to} references an unknown node`);
    }
    incoming.get(e.to)!.push(e.from);
  }
  const layers = new Map<string, number>();
  const visiting = new Set<string>();
  const visit = (id: string): number => {
    const known = layers.get(id);
    if (known !== undefined) return known;
    if (visiting.has(id)) throw new Error(`flow has a cycle through ${id}`);
    visiting.add(id);
    const parents = incoming.get(id)!;
    const layer = Math.max(minLayer.get(id)!, parents.length ? Math.max(...parents.map(visit)) + 1 : 0);
    visiting.delete(id);
    layers.set(id, layer);
    return layer;
  };
  flow.nodes.forEach((n) => visit(n.id));
  return layers;
}

export function layoutFlow(flow: Flow): FlowLayout {
  const layers = assignLayers(flow);
  const rows: FlowNode[][] = [];
  for (const node of flow.nodes) {
    const layer = layers.get(node.id)!;
    (rows[layer] ??= []).push(node);
  }
  const widest = Math.max(1, ...rows.map((r) => r.length));
  // Side lanes carry edges that skip a layer, so they never pass behind a node.
  const width = widest * NODE_W + (widest - 1) * GAP_X + PAD * 2 + LANE * 2;
  const placed = new Map<string, PlacedNode>();
  rows.forEach((row, layer) => {
    const rowWidth = row.length * NODE_W + (row.length - 1) * GAP_X;
    const startX = (width - rowWidth) / 2; // centred between the side lanes
    row.forEach((node, i) => {
      placed.set(node.id, {
        ...node,
        layer,
        x: startX + i * (NODE_W + GAP_X),
        y: PAD + layer * (NODE_H + GAP_Y),
      });
    });
  });
  const height = PAD * 2 + rows.length * NODE_H + (rows.length - 1) * GAP_Y;
  const edges = flow.edges.map((e): PlacedEdge => {
    const a = placed.get(e.from)!;
    const b = placed.get(e.to)!;
    const x1 = a.x + NODE_W / 2;
    const y1 = a.y + NODE_H;
    const x2 = b.x + NODE_W / 2;
    const y2 = b.y;
    if (b.layer - a.layer > 1) {
      // Long edge: out to the nearer side lane, down, then back in.
      const lane = x1 + x2 < width ? LANE / 2 : width - LANE / 2;
      const top = y1 + GAP_Y / 2;
      const bottom = y2 - GAP_Y / 2;
      return {
        ...e,
        path: `M ${x1} ${y1} C ${x1} ${top}, ${lane} ${y1}, ${lane} ${top} L ${lane} ${bottom} C ${lane} ${y2}, ${x2} ${bottom}, ${x2} ${y2 - 2}`,
        labelX: x1 + 4,
        labelY: y1 + 14,
        long: true,
      };
    }
    const mid = (y1 + y2) / 2;
    return {
      ...e,
      path: `M ${x1} ${y1} C ${x1} ${mid}, ${x2} ${mid}, ${x2} ${y2 - 2}`,
      labelX: (x1 + x2) / 2 + 4,
      labelY: mid,
    };
  });
  return { width, height, nodes: [...placed.values()], edges };
}
