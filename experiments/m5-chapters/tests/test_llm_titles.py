import pytest

from lib import llm_titles as L


def test_sample_text_keeps_short_text_whole():
    assert L.sample_text("a  b\nc", max_words=10) == "a b c"


def test_sample_text_head_plus_spread_windows():
    words = [f"w{i}" for i in range(1000)]
    out = L.sample_text(" ".join(words), max_words=100, n_windows=4)
    parts = out.split(" … ")
    assert len(parts) == 5
    assert parts[0].split() == words[:50]
    assert sum(len(p.split()) for p in parts) <= 100
    assert parts[-1].split()[-1] == "w999"  # the last window reaches the end


def test_build_messages_language_and_previous():
    m = L.build_messages("hola", "es", ["Intro", "Fútbol"])
    assert m[0]["role"] == "system" and "podcast" in m[0]["content"]
    assert "en español" in m[1]["content"] and "Intro; Fútbol" in m[1]["content"]
    assert "hola" in m[1]["content"]
    en = L.build_messages("hi", "en")
    assert "in English" in en[1]["content"] and "already used" not in en[1]["content"]
    assert L.build_messages("x", "fr")[1]["content"] == L.build_messages("x", "en")[1]["content"]


def test_render_prompt_passes_enable_thinking_and_bos():
    tpl = ("{{ bos_token }}{% for m in messages %}<{{ m.role }}>{{ m.content }}{% endfor %}"
           "{% if add_generation_prompt %}<assistant>{% if not enable_thinking %}<think></think>{% endif %}{% endif %}")
    out = L.render_prompt(tpl, [{"role": "user", "content": "hi"}], bos="<s>")
    assert out == "<s><user>hi<assistant><think></think>"


def test_render_prompt_raise_exception_helper():
    with pytest.raises(ValueError, match="boom"):
        L.render_prompt("{{ raise_exception('boom') }}", [])


@pytest.mark.parametrize("raw,title", [
    ("¿Qué es EC2? ¿Qué es EC2? ¿Qué es EC2?", "¿Qué es EC2?"),
    ("Cerrado Cerrado Cerrado Cerrado", "Cerrado"),
    ("a b a b c", "a b c"),
])
def test_collapse_repeats(raw, title):
    assert L.collapse_repeats(raw) == title


@pytest.mark.parametrize("raw,title", [
    ('"La crisis de la vivienda"', "La crisis de la vivienda"),
    ("Título: La crisis de la vivienda.", "La crisis de la vivienda"),
    ("Chapter 5: Advantages of remote work", "Advantages of remote work"),
    ("Capítulo 3 - Los jueces", "Los jueces"),
    ("**Los jueces y la ley**\n\nEste título resume…", "Los jueces y la ley"),
    ("<think>hmm, the topic is\nfootball</think>\nFootball tactics", "Football tactics"),
    ("<think>never closed", None),
    ("2. Road salt in lakes", "Road salt in lakes"),
    ("one two three four five six seven eight nine ten eleven twelve", "one two three four five six seven eight nine ten"),
    ("", None),
    ("  \n ", None),
])
def test_postprocess(raw, title):
    assert L.postprocess(raw) == title


def test_wrong_language_only_when_confident_other_language():
    lid = lambda conf, pred: (lambda t: (pred, conf))
    assert L.wrong_language("x", "es", lid(0.95, "en"))
    assert not L.wrong_language("x", "es", lid(0.6, "en"))   # unsure
    assert not L.wrong_language("x", "es", lid(0.99, "es"))  # right language
    assert not L.wrong_language("x", "es", lid(0.99, "it"))  # third language: short-string noise
