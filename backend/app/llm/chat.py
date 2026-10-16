"""Chat completions over the shared llama.cpp instance (docs/decisions/0011 §1).

`ChatLLM` is the interface the LLM-backed tools (show notes, relevance)
code against, so tests can pass a fake. `LlamaChat` implements it over
`app.llm.runtime`'s process-wide `Llama`, holding `llm_lock` around
every call.
"""

import json
from typing import Protocol

from app.chapters.llm_titles import render_prompt
from app.llm import runtime


class ChatLLM(Protocol):
    def count_tokens(self, text: str) -> int: ...

    def complete(self, messages: list[dict], *, max_tokens: int, temperature: float = 0.0, seed: int = 0,
                 json_schema: dict | None = None) -> str: ...


class LlamaChat:
    """`ChatLLM` over the shared llama.cpp instance; every call holds `runtime.llm_lock`."""

    def __init__(self, llm, n_ctx: int):
        self.llm, self.n_ctx = llm, n_ctx
        self.template, self.bos, self.eos = runtime.chat_tokens(llm)

    def count_tokens(self, text: str) -> int:
        with runtime.llm_lock:
            return len(self.llm.tokenize(text.encode(), add_bos=False, special=False))

    def complete(self, messages: list[dict], *, max_tokens: int, temperature: float = 0.0, seed: int = 0,
                 json_schema: dict | None = None) -> str:
        from llama_cpp import LlamaGrammar

        prompt = render_prompt(self.template, messages, self.bos, self.eos)
        grammar = LlamaGrammar.from_json_schema(json.dumps(json_schema), verbose=False) if json_schema else None
        with runtime.llm_lock:
            # The template already starts with BOS when the model wants one.
            ids = self.llm.tokenize(prompt.encode(), add_bos=False, special=True)
            if len(ids) + max_tokens > self.n_ctx:
                raise ValueError(f"prompt of {len(ids)} tokens + {max_tokens} doesn't fit n_ctx={self.n_ctx}")
            out = self.llm.create_completion(ids, max_tokens=max_tokens, temperature=temperature, seed=seed,
                                             grammar=grammar)
        return out["choices"][0]["text"]
