/**
 * Completeness and well-formedness of the explanation content
 * (docs/decisions/0021): every tool has a doc, every flow is a valid
 * DAG, every glossary key used exists, and every custom model shows
 * both its inference and its training flow.
 */
import { TOOL_TYPES } from '../jobs/job.model';
import { Flow, NODE_H, NODE_W, assignLayers, layoutFlow } from './flow';
import { GLOSSARY, GlossaryKey } from './glossary';
import { SYSTEM_FLOW, TOOL_DOCS } from './tool-docs';

const allDocs = Object.values(TOOL_DOCS);
const allFlows: [string, Flow][] = [
  ['system', SYSTEM_FLOW],
  ...allDocs.flatMap((d): [string, Flow][] => [
    [`${d.key} inference`, d.flow],
    ...(d.training ? ([[`${d.key} training`, d.training.flow]] as [string, Flow][]) : []),
  ]),
];

describe('tool docs', () => {
  it('cover every tool', () => {
    for (const tool of TOOL_TYPES) {
      expect(TOOL_DOCS[tool]?.key).toBe(tool);
    }
  });

  it.each(allDocs.map((d) => [d.key, d] as const))('%s has a one-liner, steps, inputs and outputs', (_key, doc) => {
    expect(doc.oneLiner.length).toBeGreaterThan(20);
    expect(doc.steps.length).toBeGreaterThan(0);
    expect(doc.inputs.length).toBeGreaterThan(0);
    expect(doc.outputs.length).toBeGreaterThan(0);
    expect(doc.source).toBeTruthy();
    for (const dep of doc.dependsOn) {
      expect(TOOL_TYPES).toContain(dep);
    }
  });

  it('only use glossary terms that exist', () => {
    const used = allDocs.flatMap((d) => d.steps.flatMap((s) => s.terms ?? []));
    for (const key of used) {
      expect(GLOSSARY[key as GlossaryKey], key).toBeDefined();
    }
  });

  it('show a training flow for every custom (trained or v0) model', () => {
    for (const key of ['chapters', 'relevance', 'ctr', 'hook'] as const) {
      expect(TOOL_DOCS[key].training, key).toBeDefined();
    }
    expect(TOOL_DOCS.chapters.training!.planned).toBe(false);
    expect(TOOL_DOCS.relevance.training!.planned).toBe(true);
  });

  it('state what a relative score does not mean', () => {
    for (const key of ['ctr', 'hook', 'relevance'] as const) {
      expect(TOOL_DOCS[key].metric?.doesNotMean).toBeTruthy();
    }
  });

  it('link every system-flow node to an existing doc', () => {
    for (const node of SYSTEM_FLOW.nodes) {
      expect(TOOL_DOCS[node.link as keyof typeof TOOL_DOCS], node.id).toBeDefined();
    }
  });
});

describe('flows', () => {
  it.each(allFlows)('%s is a valid DAG with unique node ids', (_name, flow) => {
    const ids = flow.nodes.map((n) => n.id);
    expect(new Set(ids).size).toBe(ids.length);
    expect(() => assignLayers(flow)).not.toThrow();
  });

  it.each(allFlows)('%s lays out without overlapping nodes', (_name, flow) => {
    const { nodes, width, height } = layoutFlow(flow);
    for (const a of nodes) {
      expect(a.x).toBeGreaterThanOrEqual(0);
      expect(a.x + NODE_W).toBeLessThanOrEqual(width);
      expect(a.y + NODE_H).toBeLessThanOrEqual(height);
      for (const b of nodes) {
        if (a === b) continue;
        const overlap = a.x < b.x + NODE_W && b.x < a.x + NODE_W && a.y < b.y + NODE_H && b.y < a.y + NODE_H;
        expect(overlap, `${a.id} overlaps ${b.id}`).toBe(false);
      }
    }
  });
});

describe('layout', () => {
  const flow: Flow = {
    nodes: [
      { id: 'a', label: 'A', kind: 'data' },
      { id: 'b', label: 'B', kind: 'process' },
      { id: 'c', label: 'C', kind: 'process' },
      { id: 'd', label: 'D', kind: 'output' },
    ],
    edges: [
      { from: 'a', to: 'b' },
      { from: 'a', to: 'c' },
      { from: 'b', to: 'd' },
      { from: 'c', to: 'd' },
    ],
  };

  it('puts branches side by side and a merge below all its inputs', () => {
    const layers = assignLayers(flow);
    expect([layers.get('a'), layers.get('b'), layers.get('c'), layers.get('d')]).toEqual([0, 1, 1, 2]);
    const placed = new Map(layoutFlow(flow).nodes.map((n) => [n.id, n]));
    expect(placed.get('b')!.y).toBe(placed.get('c')!.y);
    expect(placed.get('b')!.x).toBeLessThan(placed.get('c')!.x);
  });

  it('routes an edge that skips a layer through a side lane, clear of the nodes', () => {
    const skip: Flow = {
      nodes: [
        { id: 'a', label: 'A', kind: 'data' },
        { id: 'b', label: 'B', kind: 'process' },
        { id: 'c', label: 'C', kind: 'output' },
      ],
      edges: [
        { from: 'a', to: 'b' },
        { from: 'b', to: 'c' },
        { from: 'a', to: 'c', label: 'skip' },
      ],
    };
    const { edges, nodes } = layoutFlow(skip);
    const long = edges.find((e) => e.from === 'a' && e.to === 'c')!;
    expect(long.long).toBe(true);
    expect(edges.find((e) => e.from === 'a' && e.to === 'b')!.long).toBeUndefined();
    const laneX = Number(/ L ([\d.]+) /.exec(long.path)![1]);
    for (const n of nodes) {
      expect(laneX < n.x || laneX > n.x + NODE_W).toBe(true);
    }
  });

  it('honours minLayer', () => {
    const late: Flow = { nodes: [{ id: 'x', label: 'X', kind: 'data', minLayer: 3 }], edges: [] };
    expect(assignLayers(late).get('x')).toBe(3);
  });

  it('rejects unknown nodes and cycles', () => {
    expect(() => assignLayers({ nodes: [{ id: 'a', label: 'A', kind: 'data' }], edges: [{ from: 'a', to: 'z' }] })).toThrow(
      /unknown node/,
    );
    expect(() =>
      assignLayers({
        nodes: [
          { id: 'a', label: 'A', kind: 'data' },
          { id: 'b', label: 'B', kind: 'data' },
        ],
        edges: [
          { from: 'a', to: 'b' },
          { from: 'b', to: 'a' },
        ],
      }),
    ).toThrow(/cycle/);
  });
});
