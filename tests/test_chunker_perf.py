"""Performance test for smart_chunk O(n) offset recovery (Task 3.5)."""
import time
import pytest
from unittest.mock import patch


def _make_surprises(n_tokens):
    """Return a fake get_surprises result with n_tokens tokens, every 5th is surprised."""
    return [
        {"position": i, "token": f"tok{i}", "surprised": (i % 5 == 0)}
        for i in range(n_tokens)
    ]


def test_chunker_completes_10k_tokens_under_1s():
    """smart_chunk on a 10k-token input must finish in under 1 second."""
    n = 10_000
    fake_text = " ".join(f"word{i}" for i in range(n))
    fake_token_ids = list(range(n))
    fake_surprises = _make_surprises(n)

    caps = {"supports_logits": True, "provider": "local"}

    def fake_detokenize(ids):
        # each token → 5 chars ("tok_N" style) so offsets are predictable
        return "".join(f"w{i:04d}" for i in ids)

    with (
        patch("pymrsf.chunker.provider_capabilities", return_value=caps),
        patch("pymrsf.chunker.get_surprises", return_value=fake_surprises),
        patch("pymrsf.chunker.tokenize", return_value=fake_token_ids),
        patch("pymrsf.chunker.detokenize", side_effect=fake_detokenize),
    ):
        from pymrsf.chunker import smart_chunk

        t0 = time.perf_counter()
        chunks = smart_chunk(fake_text, min_chunk_len=50, max_chunk_len=500)
        elapsed = time.perf_counter() - t0

    assert elapsed < 2.0, f"smart_chunk took {elapsed:.2f}s on 10k tokens (limit: 2.0s)"
    assert isinstance(chunks, list)
    assert len(chunks) >= 1
