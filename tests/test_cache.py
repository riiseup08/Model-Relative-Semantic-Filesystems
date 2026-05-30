"""Tests for cache.py — LRU embedding cache and score cache behaviour."""
import time
import pytest
from unittest.mock import patch


def _fresh_cache():
    """Re-import cache with a clean state."""
    import importlib
    import pymrsf.cache as c
    importlib.reload(c)
    return c


def test_lru_evicts_oldest_on_full():
    c = _fresh_cache()
    c.configure_cache(embedding_max_size=3)
    c.set_cached_embedding("a", [1.0])
    c.set_cached_embedding("b", [2.0])
    c.set_cached_embedding("c", [3.0])
    # Cache is now full (3/3); adding "d" should evict "a" (LRU)
    c.set_cached_embedding("d", [4.0])
    assert c.get_cached_embedding("a") is None, "oldest entry should be evicted"
    assert c.get_cached_embedding("d") is not None


def test_access_updates_lru_order():
    c = _fresh_cache()
    c.configure_cache(embedding_max_size=3)
    c.set_cached_embedding("a", [1.0])
    c.set_cached_embedding("b", [2.0])
    c.set_cached_embedding("c", [3.0])
    # Access "a" to make it recently used
    c.get_cached_embedding("a")
    # Add "d" — should evict "b" (now the LRU), not "a"
    c.set_cached_embedding("d", [4.0])
    assert c.get_cached_embedding("b") is None, "b should be evicted (LRU after 'a' was accessed)"
    assert c.get_cached_embedding("a") is not None, "a should survive (recently accessed)"


def test_evictions_counter_increments():
    c = _fresh_cache()
    c.configure_cache(embedding_max_size=2)
    c.set_cached_embedding("x", [1.0])
    c.set_cached_embedding("y", [2.0])
    c.set_cached_embedding("z", [3.0])  # triggers eviction
    stats = c.get_embedding_cache_stats()
    assert stats["evictions"] == 1
    assert stats["clears"] == 1  # backward-compat alias


def test_ttl_expiry():
    c = _fresh_cache()
    c.configure_cache(embedding_ttl=1)  # 1 second TTL
    c.set_cached_embedding("expire_me", [5.0])
    with patch("pymrsf.cache.time") as mock_time:
        mock_time.time.return_value = time.time() + 5  # advance 5 seconds
        result = c.get_cached_embedding("expire_me")
    assert result is None, "entry should be expired after TTL"


def test_cache_hit_on_same_chunk():
    import pymrsf.cache as c
    from unittest.mock import patch as _patch
    with _patch("pymrsf.rag.provider_capabilities") as mock_caps, \
         _patch("pymrsf.rag.embed", return_value=[0.1] * 768), \
         _patch("pymrsf.rag.probe", None):
        mock_caps.return_value = {"supports_probe": False, "supports_embeddings": True,
                                  "supports_delta": False, "provider": "openai"}
        from pymrsf.rag import score_chunk
        c.clear_cache(reset_stats=True)
        score_chunk("hello world", query="test", use_cache=True)
        result2 = score_chunk("hello world", query="test", use_cache=True)
    assert result2.get("cached") is True


def test_provider_key_separation():
    """Scores cached under different providers should not cross-hit."""
    import pymrsf.cache as c
    fake_result_a = {"rag_score": 77, "verdict": "good"}
    fake_result_b = {"rag_score": 33, "verdict": "weak"}
    c.set_cached_score("chunk", "query", {}, fake_result_a, provider="openai", model_version="v1")
    c.set_cached_score("chunk", "query", {}, fake_result_b, provider="anthropic", model_version="v1")
    hit_a = c.get_cached_score("chunk", "query", {}, provider="openai", model_version="v1")
    hit_b = c.get_cached_score("chunk", "query", {}, provider="anthropic", model_version="v1")
    assert hit_a["rag_score"] == 77
    assert hit_b["rag_score"] == 33


def test_clear_resets_evictions_when_requested():
    c = _fresh_cache()
    c.configure_cache(embedding_max_size=1)
    c.set_cached_embedding("p", [1.0])
    c.set_cached_embedding("q", [2.0])  # evicts "p"
    c.clear_embedding_cache(reset_stats=True)
    stats = c.get_embedding_cache_stats()
    assert stats["evictions"] == 0
    assert stats["clears"] == 0
