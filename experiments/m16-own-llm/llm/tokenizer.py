"""Our tokenizer (0025 §4): byte-level BPE, trained from scratch with HF `tokenizers`.

The pre-tokenizer is the Llama-3 split regex followed by GPT-2 byte-level
mapping -- exactly the pipeline llama.cpp implements as its `llama-bpe`
pre-tokenizer, so the GGUF export tokenizes identically (checked by the export
parity test at stage A).

Chat format: ChatML-style, rendered by a jinja template that is stored in the
GGUF metadata, so the app's `render_prompt` works unchanged. The template
ignores `enable_thinking` (the model has no thinking mode).
"""

from collections.abc import Iterable
from pathlib import Path

from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers

EOT, IM_START, IM_END, PAD = "<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|pad|>"
SPECIALS = [EOT, IM_START, IM_END, PAD]

# Llama-3's split pattern (tiktoken cl100k-style), as in its tokenizer.json.
LLAMA3_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+|\s+(?!\S)|\s+"
)

CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


def new_tokenizer() -> Tokenizer:
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(LLAMA3_PATTERN), behavior="isolated", invert=False),
        pre_tokenizers.ByteLevel(add_prefix_space=False, trim_offsets=False, use_regex=False),
    ])
    tok.decoder = decoders.ByteLevel()
    return tok


def train_tokenizer(texts: Iterable[str], vocab_size: int, out_path: str | Path | None = None,
                    min_frequency: int = 2) -> Tokenizer:
    tok = new_tokenizer()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=min_frequency, special_tokens=SPECIALS,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    tok.train_from_iterator(texts, trainer)
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        tok.save(str(out_path))
    return tok


def load_tokenizer(path: str | Path) -> Tokenizer:
    return Tokenizer.from_file(str(Path(path).expanduser()))


def render_chat(messages: list[dict], add_generation_prompt: bool = False) -> str:
    """Same jinja settings as the backend's `render_prompt` (app/chapters/llm_titles.py)."""
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    return env.from_string(CHAT_TEMPLATE).render(messages=messages, add_generation_prompt=add_generation_prompt,
                                                 enable_thinking=False, bos_token="", eos_token="")
