"""An own-model GGUF (M16) through the app's real llama.cpp path: registry, chat template from the
GGUF metadata, BOS/EOS, a plain completion, a grammar-constrained JSON completion and the titler.

The fixture (tests/fixtures/own_llm, ~80 KB, random weights) is written by our real export path:
experiments/m16-own-llm/tools/make_backend_fixture.py. Its text is gibberish; only the plumbing is
checked. See docs/decisions/0025-own-llm-m16.md §10-11."""

import json
from pathlib import Path

import pytest

from app.config import get_settings
from app.llm import registry, runtime
from app.llm.chat import LlamaChat

FIX = Path(__file__).parent / "fixtures" / "own_llm"
GGUF = FIX / "yts-own-tiny.gguf"


@pytest.fixture
def own(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(get_settings(), "llm_own_dir", FIX)
    llm = runtime.get_llama(GGUF, 512, 2, n_gpu_layers=0)
    yield llm
    runtime.release_all()


def test_registry_resolves_the_fixture(own) -> None:
    spec = registry.resolve("own")
    assert spec.path == GGUF and spec.n_ctx == 512 and spec.label == "yts-own-tiny"


def test_chat_tokens_from_own_gguf(own) -> None:
    template, bos, eos = runtime.chat_tokens(own)
    assert "<|im_start|>" in template
    assert bos == ""  # the GGUF declares add_bos_token = false
    assert eos == "<|im_end|>"


def test_chat_completion_and_json_grammar(own) -> None:
    chat = LlamaChat(own, 512)
    msgs = [{"role": "system", "content": "Responde solo con JSON."}, {"role": "user", "content": "Hola"}]
    assert isinstance(chat.complete(msgs, max_tokens=12), str)
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"],
              "additionalProperties": False}
    out = chat.complete(msgs, max_tokens=120, json_schema=schema)
    assert json.loads(out).keys() == {"ok"}  # the grammar holds a random model to the schema


def test_titler_over_own_gguf(own) -> None:
    from app.chapters.llm_titles import LlamaTitler

    t = LlamaTitler(GGUF, n_ctx=512, max_tokens=16, max_words=60, llm=own)
    r = t.title("Hablamos de la montaña y de la canción del episodio.", "es")
    assert r.attempts in (1, 2) and r.prompt_tokens > 0  # a gibberish title may be rejected: plumbing only
