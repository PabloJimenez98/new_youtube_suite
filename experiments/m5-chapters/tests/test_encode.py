import numpy as np

from lib.encode import Encoder


def test_length_sorting_preserves_input_order():
    enc = Encoder("minilm", device="cpu")
    texts = ["a much longer sentence about something else entirely", "hi", "medium length text"]
    together = enc(texts, batch_size=2)
    one_by_one = np.vstack([enc([t]) for t in texts])
    assert np.allclose(together, one_by_one, atol=1e-4)
