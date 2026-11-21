import { Component } from '@angular/core';

import { Chart } from '../../charts/chart';
import { ChartCard, ChartHelp } from '../../charts/chart-card';
import { StatTile } from '../../dashboard/stat-tile/stat-tile';
import { GLOSSARY, GlossaryKey } from '../../explain/glossary';
import { TOOL_DOCS } from '../../explain/tool-docs';
import { FlowDiagram } from '../../shared/flow-diagram/flow-diagram';
import { InfoButton } from '../../shared/info-button/info-button';
import { OwnLlmArch } from '../own-llm-arch/own-llm-arch';
import { Term } from '../../shared/term/term';
import {
  GATES,
  ablations,
  dpoAcc,
  dpoLoss,
  gateRates,
  gateSpearman,
  lrSchedule,
  paramBreakdown,
  SPEARMAN_GATE,
  passes,
  pretrainBpb,
  pretrainLoss,
  scaling,
  sftGeneral,
  sftTasks,
  stageTokens,
} from '../own-llm-charts';
import { OWN_LLM_DATA } from '../own-llm-data';

/** Qwen file measured on disk (backend/models/chapters/.../Qwen3.5-9B-Q4_K_M.gguf). */
const QWEN_GGUF_BYTES = 5_680_522_464;
const OWN_GGUF_BYTES = 202_595_328;

interface Verdict {
  task: string;
  ok: boolean;
  headline: string;
  detail: string;
}

/**
 * How it works → "Our own language model" (M16, decision 0025): what the
 * model is, how it was built and trained, and how it did, from the measured
 * runs only (own-llm-data.ts is generated from metrics.jsonl and gates.json).
 */
@Component({
  selector: 'app-own-llm-section',
  imports: [Chart, ChartCard, StatTile, FlowDiagram, InfoButton, Term, OwnLlmArch],
  templateUrl: './own-llm-section.html',
  styleUrl: './own-llm-section.scss',
})
export class OwnLlmSection {
  protected readonly d = OWN_LLM_DATA;
  protected readonly trainingFlow = TOOL_DOCS.ownllm.training!.flow;
  protected readonly gates = GATES;
  protected readonly passes = passes;

  protected readonly charts = {
    params: paramBreakdown(),
    stages: stageTokens(),
    loss: pretrainLoss(),
    bpb: pretrainBpb(),
    lr: lrSchedule(),
    scaling: scaling(),
    ablations: ablations(),
    sftGeneral: sftGeneral(),
    sftTasks: sftTasks(),
    dpoLoss: dpoLoss(),
    dpoAcc: dpoAcc(),
    gateRates: gateRates(),
    gateSpearman: gateSpearman(),
  };

  private readonly p = OWN_LLM_DATA.arch.params;
  private readonly hours = OWN_LLM_DATA.arch.pretrainTokens / OWN_LLM_DATA.pretrain.tokS / 3600;
  private readonly finalBpb = OWN_LLM_DATA.pretrain.val[OWN_LLM_DATA.pretrain.val.length - 1][1];

  protected readonly tiles = [
    { label: 'Parameters', value: `${(this.p.total / 1e6).toFixed(1)}M`, hint: `Qwen3.5-9B has ~9B: about ${Math.round(9e9 / this.p.total)}× more` },
    { label: 'Pretraining tokens', value: `${this.d.arch.pretrainTokens / 1e9}B`, hint: `about ${Math.round(this.hours)} h on the RTX 5070` },
    { label: 'Model file', value: `${Math.round(OWN_GGUF_BYTES / 1e6)} MB`, hint: `Qwen’s is ${(QWEN_GGUF_BYTES / 1e9).toFixed(2)} GB (${Math.round(QWEN_GGUF_BYTES / OWN_GGUF_BYTES)}× larger)` },
    { label: 'Held-out bits per byte', value: this.finalBpb.toFixed(3), hint: 'lower is better · 1.37 after the first 0.25B tokens' },
    { label: 'Tasks that pass', value: '1 of 3', hint: 'titles pass · show notes and relevance do not' },
  ];

  protected readonly verdicts: Verdict[] = [
    {
      task: 'Chapter titles',
      ok: true,
      headline: 'Usable',
      detail: `No fallbacks and ${(this.d.gates['own-dpo'].titlesLang * 100).toFixed(1)} % in the right language on 49 unseen episodes.`,
    },
    {
      task: 'Show notes',
      ok: false,
      headline: 'Use Qwen',
      detail: `${Math.round(this.d.gates['own-dpo'].shownotesValid * 100)} % of drafts valid (gate 95 %): repeated tags, a few truncated JSON answers.`,
    },
    {
      task: 'Relevance',
      ok: false,
      headline: 'Use Qwen',
      detail: 'Answers always parse, but the scores are smooth shapes that ignore what is said (Spearman −0.07 vs Qwen).',
    },
  ];

  protected readonly facts: [string, string][] = [
    ['Type', 'Decoder-only transformer (Llama-style)'],
    ['Width × depth', `${this.d.arch.d_model} × ${this.d.arch.n_layers} blocks`],
    ['Attention heads', `${this.d.arch.n_heads} query / ${this.p.kvHeads} key-value, ${this.p.headDim} dims each`],
    ['Context', `${this.d.arch.max_seq_len.toLocaleString('en-US')} tokens (pretrained at ${this.d.arch.seqLen.toLocaleString('en-US')})`],
    ['Text mix', `${this.d.arch.mix.es * 100} % Spanish · ${this.d.arch.mix.en * 100} % English`],
    ['Weights', 'random at the start: nothing pretrained, the tokenizer included'],
  ];

  protected readonly comparison: [string, string, string][] = [
    ['Parameters', `${(this.p.total / 1e6).toFixed(1)}M`, '~9B'],
    ['Trained by', 'us, from random weights', 'Alibaba (pretrained, open weights)'],
    ['Pretraining text', `${this.d.arch.pretrainTokens / 1e9}B tokens`, 'trillions of tokens'],
    ['Vocabulary', '32,768 tokens', '~248k tokens'],
    ['Tokens per Spanish word', '1.347 (7 % fewer)', '1.449'],
    ['Model file', `${Math.round(OWN_GGUF_BYTES / 1e6)} MB (f16)`, `${(QWEN_GGUF_BYTES / 1e9).toFixed(2)} GB (4-bit Q4_K_M)`],
    ['Role in the app', 'selectable experiment', 'default for every LLM tool'],
  ];

  protected readonly spearmanSubtitle = `dashed = gate ${SPEARMAN_GATE.toFixed(2)} · dotted = Qwen vs itself ${OWN_LLM_DATA.gates['own-dpo'].qwenPassToPass.toFixed(2)}`;

  /** Per-stage summary (configs/a3/*.yaml and the runs' logs; decision 0025 "A2 and A3 results"). */
  protected readonly stages = [
    {
      name: 'Pretraining',
      what: 'Predict the next token of web text, from random weights.',
      data: `${this.d.arch.pretrainTokens / 1e9}B tokens: FineWeb2-HQ (es) + FineWeb-Edu (en)`,
      lr: `${this.d.arch.lr} · warm-up, constant, decay over the last 20 %`,
      result: `held-out bpb ${this.finalBpb.toFixed(3)}`,
    },
    {
      name: 'General SFT',
      what: 'Answer instead of continuing: the chat format.',
      data: `${Math.round(this.d.post.tokens['a3-sft-general'] / 1e6)}M tokens of open instruction data (es + en)`,
      lr: '3e-4 · one epoch at 4,096 tokens of context',
      result: `validation loss ${this.d.post.sftGeneral[0][1].toFixed(2)} → ${this.d.post.sftGeneral[this.d.post.sftGeneral.length - 1][1].toFixed(2)}`,
    },
    {
      name: 'Task SFT',
      what: 'Copy Qwen on our three tasks, with the app’s exact prompts.',
      data: '13,498 examples from 259 episodes, plus general replay (15 % of the loss tokens)',
      lr: '1.5e-4 · one epoch (three overfitted)',
      result: `validation loss ${this.d.post.sftTasks[this.d.post.sftTasks.length - 1][1].toFixed(2)}`,
    },
    {
      name: 'DPO',
      what: 'Prefer Qwen’s answer over our own answers that fail a check.',
      data: '1,509 preference pairs (bad JSON, too-long titles, wrong ranking)',
      lr: '2e-6 · β 0.1 · two epochs',
      result: `preference accuracy ${Math.round(this.d.post.dpoAcc[this.d.post.dpoAcc.length - 1][1] * 100)} %`,
    },
  ];

  protected readonly terms: GlossaryKey[] = ['transformer', 'bpe', 'pretraining', 'bpb', 'gqa', 'ablation', 'sft', 'dpo', 'spearman'];
  protected readonly glossary = GLOSSARY;

  protected readonly help: Record<string, ChartHelp> = {
    params: {
      what: 'Where the model’s 100.7M numbers (weights) are.',
      method: 'Counted from the stage A config, exactly as the PyTorch model is built (and checked against it).',
      read: 'Most weights are in the MLPs. A quarter is the vocabulary table, which is used twice: to read tokens in and to score the next token.',
    },
    stages: {
      what: 'How much text each training stage processed.',
      method: 'Token counts logged by each run (metrics.jsonl). Log scale.',
      read: 'Pretraining is where the model learns the language; the later stages are small in comparison and only teach it how to answer.',
    },
    loss: {
      what: 'How well the model predicts the next token of the text it is training on, during pretraining.',
      method: 'Cross-entropy loss in nats per token, logged every 10 steps (one step = 524,288 tokens), here sub-sampled.',
      read: 'It falls fast at first (learning words and grammar) and slowly afterwards. The last drop is the learning-rate decay.',
    },
    bpb: {
      what: 'The same, but on held-out text the model never trains on, per language.',
      method: 'Bits per byte on fixed validation text, every 250M tokens.',
      read: 'Lower is better. Spanish is easier for the model than English because it saw 70 % Spanish. No sign of over-fitting: train and held-out both keep improving.',
    },
    lr: {
      what: 'The learning rate: how big each training step is.',
      method: 'Warm-up over the first 100M tokens, constant at 2e-3, then a linear decay to 0 over the last 20 % (the shaded area).',
      read: 'The decay is where the model “settles”: the loss drops visibly there.',
    },
    scaling: {
      what: 'Held-out bpb of the same recipe trained on 0.5, 1, 2 and 5 billion tokens.',
      method: 'Four separate runs; the 1B point is the chosen ablation. Log x-axis.',
      read: 'Each doubling of data still helps, by less each time. More text (or a larger model) would keep improving it.',
    },
    ablations: {
      what: 'Seven changes to the baseline recipe, each tried alone on 1B tokens.',
      method: 'Change in held-out bpb against the baseline. Blue = better, red = worse. A change was kept only if it helped clearly at no throughput cost.',
      read: 'Only the higher learning rate was adopted (✓). Muon was slightly better still, but by too little to justify a new optimizer; the cosine schedule was clearly worse.',
    },
    sftGeneral: {
      what: 'General SFT: validation loss on held-out conversations.',
      method: 'One epoch of 244M tokens of open instruction data, 4,096-token context.',
      read: 'Steady improvement: the model learns the chat format and to answer instead of continuing text.',
    },
    sftTasks: {
      what: 'Task SFT: validation loss on held-out episodes of our three tasks.',
      method: 'Targets are Qwen’s answers on 259 episodes. One epoch (a 3-epoch run overfitted and was discarded).',
      read: 'Short and small: the tasks share a few prompt formats, so one pass is enough.',
    },
    dpoLoss: {
      what: 'DPO loss on held-out answer pairs.',
      method: 'It starts at ln 2 ≈ 0.69 by construction (the model has no preference yet) and falls as it prefers the good answer.',
      read: 'Lower means the model now favours the preferred answer more strongly.',
    },
    dpoAcc: {
      what: 'How often the model gives the preferred answer a higher probability than the rejected one.',
      method: 'On 123 held-out pairs.',
      read: '85 %: it learnt the preference. That improved show notes, but not relevance (see the gates).',
    },
    gateRates: {
      what: 'The pass/fail checks on 49 test episodes, before and after DPO.',
      method: 'The app’s own pipelines run with our model; the white tick is each gate’s threshold.',
      read: 'A bar past its tick passes. DPO lifted show notes from 57 % to 86 %, short of 95 %.',
    },
    gateSpearman: {
      what: 'Whether our relevance scores rank moments in the same order as Qwen’s.',
      method: 'Spearman correlation per episode, averaged. The gate is 80 % of how well Qwen agrees with itself across two passes (0.49).',
      read: 'Near zero: our model’s scores do not follow the content. This is the clearest gap to Qwen.',
    },
  };
}
