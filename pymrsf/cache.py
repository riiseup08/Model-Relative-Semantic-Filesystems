"""
pymrsf.cache — Caching layer for RAG chunk scores

Avoids re-scoring the same chunks across multiple queries.
Thread-safe, LRU caches with statistics.

Caching strategy:
  - Score cache: LRU eviction via OrderedDict.popitem(last=False).
    move_to_end() on every hit keeps hot entries alive.
  - Embedding cache: same LRU strategy, separate size/TTL limits.

Threading model:
  - Both caches hold their lock only for dict operations.
  - deepcopy is performed outside the lock to avoid serialising callers
    on potentially large result objects.
"""

import copy
import hashlib
import threading
import time
from collections import OrderedDict
from typing import Any

# ── Cache configuration ────────────────────────────────────────────────────────

_CACHE_ENABLED = True
_CACHE_MAX_SIZE = 10000  # Maximum number of cached scores
_CACHE_TTL = 3600  # Time-to-live in seconds (1 hour default)
_EMBEDDING_CACHE_MAX_SIZE = 5000  # Separate limit for embeddings
_EMBEDDING_CACHE_TTL = 7200  # Longer TTL for embeddings (2 hours)

# Global cache and statistics
_cache: OrderedDict = OrderedDict()  # key -> (result, timestamp), LRU order
_cache_lock = threading.Lock()
_cache_stats = {
    "hits": 0,
    "misses": 0,
    "evictions": 0,
}
# Stats are cumulative across clear operations by default
# Call reset_cache_stats() explicitly to zero out counters


def configure_cache(
    enabled: bool = True,
    max_size: int = 10000,
    ttl: int = 3600,
    embedding_max_size: int = 5000,
    embedding_ttl: int = 7200
):
    """Configure the RAG score and embedding caches.

    Both caches use LRU eviction and are thread-safe. Call this at application
    startup before any scoring calls. By default both caches are enabled with
    a 1-hour TTL for scores and 2-hour TTL for embeddings.

    Args:
        enabled: Enable/disable both caches
        max_size: Maximum score cache entries (default 10,000)
        ttl: Score TTL in seconds (0 = no expiration)
        embedding_max_size: Maximum embedding cache entries (default 5,000)
        embedding_ttl: Embedding TTL in seconds (0 = no expiration)

    Example:
        >>> configure_cache(enabled=True, max_size=5000, ttl=1800)
    """
    global _CACHE_ENABLED, _CACHE_MAX_SIZE, _CACHE_TTL, _EMBEDDING_CACHE_MAX_SIZE, _EMBEDDING_CACHE_TTL
    _CACHE_ENABLED = enabled
    _CACHE_MAX_SIZE = max_size
    _CACHE_TTL = ttl
    _EMBEDDING_CACHE_MAX_SIZE = embedding_max_size
    _EMBEDDING_CACHE_TTL = embedding_ttl


def _make_cache_key(
    chunk: str,
    query: str | None,
    weights: dict | None,
    provider: str | None = None,
    model_version: str | None = None
) -> str:
    """
    Generate a deterministic cache key from chunk, query, weights, provider, and model.

    Args:
        chunk: The text chunk
        query: The query (optional)
        weights: Scoring weights (optional)
        provider: Provider name (optional, for cross-provider differentiation)
        model_version: Model version (optional, for cross-model differentiation)

    Returns:
        SHA256 hash as cache key
    """
    # Normalize weights to ensure consistent hashing
    if weights:
        weights_str = str(sorted(weights.items()))
    else:
        weights_str = ""

    cache_input = f"{chunk}|{query or ''}|{weights_str}|{provider or ''}|{model_version or ''}"
    return hashlib.sha256(cache_input.encode()).hexdigest()


def get_cached_score(
    chunk: str,
    query: str | None = None,
    weights: dict | None = None,
    provider: str | None = None,
    model_version: str | None = None
) -> dict[str, Any] | None:
    """
    Retrieve a cached score if available and not expired.

    Args:
        chunk: The text chunk
        query: The query (optional)
        weights: Scoring weights (optional)
        provider: Provider name (optional)
        model_version: Model version (optional)

    Returns:
        Deep-copied cached result dict or None if not found/expired
    """
    if not _CACHE_ENABLED:
        return None

    key = _make_cache_key(chunk, query, weights, provider, model_version)

    cached_result = None
    with _cache_lock:
        if key in _cache:
            result, timestamp = _cache[key]

            if _CACHE_TTL > 0 and (time.time() - timestamp) > _CACHE_TTL:
                del _cache[key]
                _cache_stats["misses"] += 1
            else:
                _cache.move_to_end(key)   # mark as recently used
                _cache_stats["hits"] += 1
                cached_result = result    # ref only — deepcopy outside lock
        else:
            _cache_stats["misses"] += 1

    # deepcopy outside the lock so we don't block other threads
    return copy.deepcopy(cached_result) if cached_result is not None else None


def set_cached_score(
    chunk: str,
    query: str | None,
    weights: dict | None,
    result: dict[str, Any],
    provider: str | None = None,
    model_version: str | None = None
):
    """
    Store a score result in the cache.

    Args:
        chunk: The text chunk
        query: The query (optional)
        weights: Scoring weights (optional)
        result: The scoring result to cache
        provider: Provider name (optional)
        model_version: Model version (optional)
    """
    if not _CACHE_ENABLED:
        return

    key = _make_cache_key(chunk, query, weights, provider, model_version)
    # deepcopy before acquiring the lock so we don't hold it during allocation
    stored = (copy.deepcopy(result), time.time())

    with _cache_lock:
        if len(_cache) >= _CACHE_MAX_SIZE and key not in _cache:
            _cache.popitem(last=False)  # evict least recently used — O(1)
            _cache_stats["evictions"] += 1
        _cache[key] = stored
        _cache.move_to_end(key)  # newest entry is most recently used


def clear_cache(reset_stats: bool = False):
    """Clear all cached scores.

    By default, statistics remain cumulative across clears. Pass
    reset_stats=True to zero out the counters as well.

    Args:
        reset_stats: If True, also reset cache statistics counters

    Example:
        >>> clear_cache()
        >>> clear_cache(reset_stats=True)
    """
    with _cache_lock:
        _cache.clear()
        if reset_stats:
            _cache_stats["hits"] = 0
            _cache_stats["misses"] = 0
            _cache_stats["evictions"] = 0


def get_cache_stats() -> dict[str, Any]:
    """Get score cache statistics.

    Returns:
        Dictionary with keys: hits, misses, evictions, size, max_size,
        hit_rate (percentage), enabled

    Example:
        >>> stats = get_cache_stats()
        >>> stats["hit_rate"]
        85.0
    """
    with _cache_lock:
        total = _cache_stats["hits"] + _cache_stats["misses"]
        hit_rate = _cache_stats["hits"] / total if total > 0 else 0.0

        return {
            "hits": _cache_stats["hits"],
            "misses": _cache_stats["misses"],
            "evictions": _cache_stats["evictions"],
            "size": len(_cache),
            "max_size": _CACHE_MAX_SIZE,
            "hit_rate": round(hit_rate * 100, 2),
            "enabled": _CACHE_ENABLED,
        }


def reset_cache_stats():
    """Reset score cache statistics counters to zero.

    After calling this, hits, misses, and evictions all report 0
    until the next cache operation.

    Example:
        >>> reset_cache_stats()
        >>> get_cache_stats()["hits"]
        0
    """
    with _cache_lock:
        _cache_stats["hits"] = 0
        _cache_stats["misses"] = 0
        _cache_stats["evictions"] = 0


def print_cache_stats():
    """Print a formatted cache statistics report."""
    stats = get_cache_stats()

    print(f"\n{'═' * 60}")
    print("  PYMRSF CACHE STATISTICS")
    print(f"{'═' * 60}")
    print(f"  Status      : {'Enabled' if stats['enabled'] else 'Disabled'}")
    print(f"  Cache size  : {stats['size']:,} / {stats['max_size']:,}")
    print(f"  Hits        : {stats['hits']:,}")
    print(f"  Misses      : {stats['misses']:,}")
    print(f"  Evictions   : {stats['evictions']:,}")
    print(f"  Hit rate    : {stats['hit_rate']:.1f}%")
    print(f"{'═' * 60}\n")


# ── Embedding cache (for relevance scoring) ────────────────────────────────────

def _cached_text_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


_embedding_cache: OrderedDict = OrderedDict()  # hash -> (embedding, timestamp), LRU order
_embedding_lock = threading.Lock()
_embedding_stats = {
    "hits": 0,
    "misses": 0,
    "evictions": 0,  # Individual LRU evictions (replaces old "clears" counter)
    "clears": 0,     # Backward-compat alias — same counter as evictions
}


def get_cached_embedding(text: str):
    """
    Get cached embedding for text if available and not expired.

    Args:
        text: Text to look up

    Returns:
        Cached embedding or None if not found/expired
    """
    if not _CACHE_ENABLED:
        return None

    text_hash = _cached_text_hash(text)
    with _embedding_lock:
        if text_hash in _embedding_cache:
            embedding, timestamp = _embedding_cache[text_hash]

            if _EMBEDDING_CACHE_TTL > 0 and (time.time() - timestamp) > _EMBEDDING_CACHE_TTL:
                del _embedding_cache[text_hash]
                _embedding_stats["misses"] += 1
                return None

            _embedding_cache.move_to_end(text_hash)  # mark as recently used
            _embedding_stats["hits"] += 1
            return embedding

        _embedding_stats["misses"] += 1
        return None


def set_cached_embedding(text: str, embedding):
    """
    Cache an embedding for text.

    Note: When the cache reaches _EMBEDDING_CACHE_MAX_SIZE, it clears
    all entries wholesale (not a true LRU eviction).

    Args:
        text: Text associated with embedding
        embedding: Embedding vector to cache
    """
    if not _CACHE_ENABLED:
        return

    text_hash = _cached_text_hash(text)
    with _embedding_lock:
        if text_hash in _embedding_cache:
            _embedding_cache.move_to_end(text_hash)
        elif len(_embedding_cache) >= _EMBEDDING_CACHE_MAX_SIZE:
            _embedding_cache.popitem(last=False)  # evict least recently used
            _embedding_stats["evictions"] += 1
            _embedding_stats["clears"] = _embedding_stats["evictions"]  # keep alias in sync
        _embedding_cache[text_hash] = (embedding, time.time())


def clear_embedding_cache(reset_stats: bool = False):
    """Clear all cached embeddings.

    By default, statistics remain cumulative across clears. Pass
    reset_stats=True to zero out the counters as well.

    Args:
        reset_stats: If True, also reset embedding cache statistics

    Example:
        >>> clear_embedding_cache()
    """
    with _embedding_lock:
        _embedding_cache.clear()
        if reset_stats:
            _embedding_stats["hits"] = 0
            _embedding_stats["misses"] = 0
            _embedding_stats["evictions"] = 0
            _embedding_stats["clears"] = 0


def get_embedding_cache_stats() -> dict[str, Any]:
    """Get embedding cache statistics.

    Returns:
        Dictionary with keys: hits, misses, evictions, clears (backward compat
        alias for evictions), size, max_size, hit_rate (percentage), enabled

    Example:
        >>> stats = get_embedding_cache_stats()
        >>> stats["size"]
        42
    """
    with _embedding_lock:
        total = _embedding_stats["hits"] + _embedding_stats["misses"]
        hit_rate = _embedding_stats["hits"] / total if total > 0 else 0.0

        return {
            "hits": _embedding_stats["hits"],
            "misses": _embedding_stats["misses"],
            "evictions": _embedding_stats["evictions"],
            "clears": _embedding_stats["evictions"],  # backward-compat alias
            "size": len(_embedding_cache),
            "max_size": _EMBEDDING_CACHE_MAX_SIZE,
            "hit_rate": round(hit_rate * 100, 2),
            "enabled": _CACHE_ENABLED,
        }
