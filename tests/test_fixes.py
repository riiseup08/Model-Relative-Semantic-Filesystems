"""Model-free regression tests for the correctness fixes.

These run anywhere (no GGUF, no Ollama, no API keys) and lock in:
  #2  weight redistribution in relevance-only mode
  #6  cache key includes relevance_cutoff and thresholds
  #1  get_embedding_dim() does not deadlock
  gate the relevance-only query-ignorance gate no longer zeroes every chunk
  optional-experimental import guard
"""
import threading

import numpy as np
import pytest

import pymrsf
from pymrsf import cache
from pymrsf.rag import WeightConfig, score_chunk


# ── #6 cache key ────────────────────────────────────────────────────────────
def test_cache_key_differentiates_on_cutoff_and_thresholds():
    cache.clear_cache(reset_stats=True)
    w = {"novelty": 1.0}
    cache.set_cached_score("c", "q", w, {"rag_score": 10}, relevance_cutoff=0.30)

    assert cache.get_cached_score("c", "q", w, relevance_cutoff=0.30)["rag_score"] == 10
    # Different cutoff must not collide with the cached entry.
    assert cache.get_cached_score("c", "q", w, relevance_cutoff=0.50) is None
    # Different thresholds table must not collide either.
    assert cache.get_cached_score("c", "q", w, relevance_cutoff=0.30,
                                  thresholds=[(80, "x", "y")]) is None


def test_cache_key_isolates_provider_and_model():
    cache.clear_cache(reset_stats=True)
    w = {"novelty": 1.0}
    cache.set_cached_score("c", "q", w, {"rag_score": 7},
                           provider="local", model_version="m1")
    assert cache.get_cached_score("c", "q", w, provider="local", model_version="m1")["rag_score"] == 7
    assert cache.get_cached_score("c", "q", w, provider="openai", model_version="m1") is None
    assert cache.get_cached_score("c", "q", w, provider="local", model_version="m2") is None


# ── #2 weight redistribution ────────────────────────────────────────────────
def test_weight_redistribution_relevance_only_sums_to_one():
    wc = WeightConfig().redistribute_for_relevance_only().to_dict()
    assert wc["query_ignorance"] == 0.0
    assert wc["novelty"] == pytest.approx(0.5)
    assert wc["relevance"] == pytest.approx(0.5)
    assert sum(wc.values()) == pytest.approx(1.0)


def test_weightconfig_normalize():
    wc = WeightConfig(novelty=2, relevance=2, query_ignorance=1).normalize().to_dict()
    assert sum(wc.values()) == pytest.approx(1.0)
    assert wc["novelty"] == pytest.approx(0.4)


# ── relevance-only path (provider without probe) ────────────────────────────
@pytest.fixture
def openai_provider():
    """Switch to a probe-less provider for the duration of the test."""
    prev = pymrsf.get_config().provider
    pymrsf.set_provider("openai")
    yield
    pymrsf.set_provider(prev)


def _seed_identical_embeddings(*texts):
    v = np.ones(8, dtype="float32")
    for t in texts:
        cache.set_cached_embedding(t, v)


def test_relevance_only_not_zeroed_by_gate(openai_provider):
    """gate bug: a query in relevance-only mode used to force rag_score=0."""
    q, c = "What is X?", "X is a thing that does Y."
    _seed_identical_embeddings(q, c)
    r = score_chunk(c, query=q, use_cache=False)
    assert not r.get("skipped_by_gate")
    assert r["rag_score"] > 0


def test_relevance_only_uses_redistributed_weights(openai_provider):
    """#2: perfect relevance should reach 100, not be capped at ~80."""
    q, c = "What is X?", "X is a thing that does Y."
    _seed_identical_embeddings(q, c)  # identical vectors -> cosine 1 -> relevance 100
    r = score_chunk(c, query=q, use_cache=False)
    assert r["relevance_score"] == 100
    assert r["rag_score"] == 100
    assert r["weights_used"]["query_ignorance"] == 0.0


# ── #1 deadlock ─────────────────────────────────────────────────────────────
def test_get_embedding_dim_does_not_deadlock(monkeypatch):
    import pymrsf.embeddings as emb
    monkeypatch.setattr(emb, "_embed_dim_cache", None, raising=False)

    result = {}

    def worker():
        result["dim"] = emb.get_embedding_dim()

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    t.join(timeout=15)
    assert not t.is_alive(), "get_embedding_dim() deadlocked"
    assert isinstance(result.get("dim"), int) and result["dim"] > 0


# ── optional experimental backend ───────────────────────────────────────────
def test_experimental_import_is_optional():
    assert isinstance(pymrsf.EXPERIMENTAL_AVAILABLE, bool)
    if not pymrsf.EXPERIMENTAL_AVAILABLE:
        with pytest.raises(ImportError):
            pymrsf.mrsf_write("x")
