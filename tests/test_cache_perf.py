"""Performance and concurrency tests for cache.py (Tasks 2.2 / 2.3)."""
import time
import threading
import pymrsf.cache as cache


def _reset():
    cache._cache.clear()
    cache._cache_stats.update({"hits": 0, "misses": 0, "evictions": 0})
    cache._CACHE_ENABLED = True
    cache._CACHE_MAX_SIZE = 10000
    cache._CACHE_TTL = 3600


def test_set_insertion_under_100us():
    """Each cache insertion at capacity must complete in under 100 µs."""
    _reset()
    cache._CACHE_MAX_SIZE = 500
    dummy_result = {"rag_score": 42, "verdict": "good"}

    # Pre-fill to capacity
    for i in range(500):
        cache.set_cached_score(f"chunk{i}", f"query{i}", None, dummy_result)

    # Measure insertions that trigger LRU eviction
    times = []
    for i in range(100):
        t0 = time.perf_counter()
        cache.set_cached_score(f"new_chunk{i}", f"q{i}", None, dummy_result)
        times.append(time.perf_counter() - t0)

    avg_us = (sum(times) / len(times)) * 1e6
    assert avg_us < 100, f"Average insertion took {avg_us:.1f} µs (> 100 µs limit)"
    _reset()


def test_get_hit_under_100us():
    """Cache hit lookup must complete in under 100 µs on average."""
    _reset()
    dummy_result = {"rag_score": 77, "verdict": "good", "data": list(range(50))}
    cache.set_cached_score("chunk", "query", None, dummy_result)

    times = []
    for _ in range(200):
        t0 = time.perf_counter()
        r = cache.get_cached_score("chunk", "query", None)
        times.append(time.perf_counter() - t0)
        assert r is not None

    avg_us = (sum(times) / len(times)) * 1e6
    assert avg_us < 100, f"Average hit lookup took {avg_us:.1f} µs (> 100 µs limit)"
    _reset()


def test_concurrent_get_set_no_exceptions():
    """8 threads doing 1000 mixed get/set ops must raise no exceptions and leave stats consistent."""
    _reset()
    cache._CACHE_MAX_SIZE = 200
    errors = []
    dummy = {"rag_score": 55}

    def worker(thread_id):
        try:
            for i in range(1000):
                key = f"c{i % 50}"
                q = f"q{thread_id}"
                cache.set_cached_score(key, q, None, dummy)
                cache.get_cached_score(key, q, None)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Exceptions in worker threads: {errors}"
    stats = cache.get_cache_stats()
    assert stats["hits"] + stats["misses"] > 0
    assert stats["size"] <= cache._CACHE_MAX_SIZE
    _reset()


def test_deepcopy_isolation():
    """Mutating a returned cache hit must not corrupt the cached value."""
    _reset()
    original = {"rag_score": 60, "tags": ["a", "b"]}
    cache.set_cached_score("x", "q", None, original)

    hit = cache.get_cached_score("x", "q", None)
    hit["tags"].append("c")  # mutate the returned copy

    hit2 = cache.get_cached_score("x", "q", None)
    assert hit2["tags"] == ["a", "b"], "Cached value was mutated by caller"
    _reset()
