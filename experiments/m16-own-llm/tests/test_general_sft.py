"""llm.general_sft: turn normalisation and per-source conversion (no network, no LID model)."""

from pathlib import Path

from llm import general_sft as g


def test_clean_turns_normalises_roles_and_strips():
    out = g.clean_turns([{"from": "system", "value": " sé breve "}, {"from": "human", "value": "hola"},
                         {"from": "gpt", "value": "¡hola! "}])
    assert out == [{"role": "system", "content": "sé breve"}, {"role": "user", "content": "hola"},
                   {"role": "assistant", "content": "¡hola!"}]


def test_clean_turns_rejects_bad_shapes():
    u, a = {"role": "user", "content": "q"}, {"role": "assistant", "content": "r"}
    assert g.clean_turns([u]) is None                       # no answer
    assert g.clean_turns([u, a, u]) is None                 # ends on the user
    assert g.clean_turns([u, u, a]) is None                 # doesn't alternate
    assert g.clean_turns([a, u, a]) is None                 # starts with the assistant
    assert g.clean_turns([u, {"role": "assistant", "content": "  "}]) is None  # empty turn
    assert g.clean_turns([{"role": "tool", "content": "x"}, a]) is None        # unknown role
    assert g.clean_turns([u, a, u, a]) is not None          # multi-turn is fine


def test_convert_aya_uses_the_dataset_language():
    ex = g.convert("aya", Path("x.parquet"), {"inputs": "¿Capital de Perú?", "targets": "Lima.", "language": "Spanish"})
    assert ex["lang"] == "es" and ex["messages"][-1] == {"role": "assistant", "content": "Lima."}
    assert g.convert("aya", Path("x.parquet"), {"inputs": "a", "targets": "b", "language": "Somali"})["lang"] == "other"


def test_convert_ultrafeedback_pairs():
    p = [{"role": "user", "content": "Explain X"}]
    r = {"chosen": p + [{"role": "assistant", "content": "good"}],
         "rejected": p + [{"role": "assistant", "content": "bad"}], "score_chosen": 8.0, "score_rejected": 3.0}
    ex = g.convert("ultrafeedback", Path("train_prefs-0.parquet"), r)
    assert ex["messages"] == p and ex["chosen"] == "good" and ex["rejected"] == "bad" and ex["split"] == "train"
    assert g.convert("ultrafeedback", Path("test_prefs-0.parquet"), r)["split"] == "val"
    same = dict(r, rejected=p + [{"role": "assistant", "content": "good"}])
    assert g.convert("ultrafeedback", Path("t.parquet"), same) is None   # identical answers: no signal


def test_convert_chat_sources_use_lid(monkeypatch):
    monkeypatch.setattr(g, "lid", lambda text: "es")
    r = {"messages": [{"role": "user", "content": "hola"}, {"role": "assistant", "content": "buenas"}]}
    assert g.convert("smoltalk2-ml", Path("x"), r)["lang"] == "es"
    r = {"conversations": [{"from": "human", "value": "hola"}, {"from": "gpt", "value": "buenas"}]}
    assert g.convert("openhermes-es", Path("x"), r)["messages"][0]["role"] == "user"
