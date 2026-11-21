import { Component } from '@angular/core';

import { OWN_LLM_DATA } from '../own-llm-data';

interface ArchNode {
  label: string;
  detail: string;
  kind: 'data' | 'param' | 'norm' | 'attn' | 'ffn' | 'add';
}

const A = OWN_LLM_DATA.arch;
const P = A.params;
const vocab = A.vocab_size.toLocaleString('en-US');

/**
 * Our own LLM as a stack, input at the top: embedding, the repeated
 * transformer block (framed, × n_layers), output head. Node colours match the
 * parameter-breakdown chart (embedding / attention / MLP).
 */
@Component({
  selector: 'app-own-llm-arch',
  template: `
    <ol class="arch" aria-label="Model architecture, from input to output">
      @for (n of input; track $index) {
        <li class="arch__node arch__node--{{ n.kind }}"><strong>{{ n.label }}</strong><span>{{ n.detail }}</span></li>
      }
      <li class="arch__block">
        <span class="arch__repeat">× {{ layers }} blocks</span>
        <ol>
          @for (n of block; track $index) {
            <li class="arch__node arch__node--{{ n.kind }}"><strong>{{ n.label }}</strong><span>{{ n.detail }}</span></li>
          }
        </ol>
      </li>
      @for (n of output; track $index) {
        <li class="arch__node arch__node--{{ n.kind }}"><strong>{{ n.label }}</strong><span>{{ n.detail }}</span></li>
      }
    </ol>
  `,
  styleUrl: './own-llm-arch.scss',
})
export class OwnLlmArch {
  protected readonly layers = A.n_layers;
  protected readonly input: ArchNode[] = [
    { label: 'Tokens in', detail: `${vocab}-token BPE vocabulary, ours`, kind: 'data' },
    { label: 'Token embedding', detail: `${vocab} × ${A.d_model}: each token becomes a vector`, kind: 'param' },
  ];
  protected readonly block: ArchNode[] = [
    { label: 'RMSNorm', detail: 'rescales the vector', kind: 'norm' },
    { label: 'Attention', detail: `${A.n_heads} query heads share ${P.kvHeads} key/value heads (GQA) · rotary positions (RoPE)`, kind: 'attn' },
    { label: '+ residual', detail: 'adds the input back', kind: 'add' },
    { label: 'RMSNorm', detail: 'rescales the vector', kind: 'norm' },
    { label: 'SwiGLU MLP', detail: `${A.d_model} → ${P.hidden} → ${A.d_model}, per token`, kind: 'ffn' },
    { label: '+ residual', detail: 'adds the input back', kind: 'add' },
  ];
  protected readonly output: ArchNode[] = [
    { label: 'RMSNorm', detail: 'final', kind: 'norm' },
    { label: 'Output head', detail: `the embedding matrix again (tied): a score for each of the ${vocab} tokens`, kind: 'param' },
    { label: 'Next token', detail: 'probabilities → constrained decoding in the app', kind: 'data' },
  ];
}
