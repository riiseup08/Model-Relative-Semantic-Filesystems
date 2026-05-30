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

import hashlib
import time
import copy
from collections import OrderedDict
from typing import Optional, Dict, Any
import threading


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
    """
    Configure the RAG score cache.

    Args:
        enabled: Enable/disable caching
        max_size: Maximum number of cached score entries
        ttl: Time-to-live for scores in seconds (0 = no expiration)
        embedding_max_size: Maximum number of cached embeddings
        embedding_ttl: Time-to-live for embeddings in seconds (0 = no expiration)
    """
    global _CACHE_ENABLED, _CACHE_MAX_SIZE, _CACHE_TTL, _EMBEDDING_CACHE_MAX_SIZE, _EMBEDDING_CACHE_TTL
    _CACHE_ENABLED = enabled
    _CACHE_MAX_SIZE = max_size
    _CACHE_TTL = ttl
    _EMBEDDING_CACHE_MAX_SIZE = embedding_max_size
    _EMBEDDING_CACHE_TTL = embedding_ttl


def _make_cache_key(
    chunk: str,
    query: Optional[str],
    weights: Optional[dict],
    provider: Optional[str] = None,
    model_version: Optional[str] = None,
    relevance_cutoff: Optional[float] = None,
    thresholds: Optional[list] = None,
) -> str:
    """
    Generate a deterministic cache key from the inputs that affect the score.

    Includes relevance_cutoff and thresholds because both change the computed
    result — omitting them returned stale scores when a caller varied either.

    Args:
        chunk: The text chunk
        query: The query (optional)
        weights: Scoring weights (optional)
        provider: Provider name (optional, for cross-provider differentiation)
        model_version: Model version (optional, for cross-model differentiation)
        relevance_cutoff: Minimum cosine similarity used for relevance (optional)
        thresholds: Verdict threshold table (optional)

    Returns:
        SHA256 hash as cache key
    """
    # Normalize weights to ensure consistent hashing
    if weights:
        weights_str = str(sorted(weights.items()))
    else:
        weights_str = ""

    cutoff_str = "" if relevance_cutoff is None else repr(relevance_cutoff)
    thresholds_str = "" if thresholds is None else str(thresholds)

    cache_input = (
        f"{chunk}|{query or ''}|{weights_str}|{provider or ''}|{model_version or ''}"
        f"|{cutoff_str}|{thresholds_str}"
    )
    return hashlib.sha256(cache_input.encode()).hexdigest()


def get_cached_score(
    chunk: str,
    query: Optional[str] = None,
    weights: Optional[dict] = None,
    provider: Optional[str] = None,
    model_version: Optional[str] = None,
    relevance_cutoff: Optional[float] = None,
    thresholds: Optional[list] = None,
) -> Optional[Dict[str, Any]]:
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

    key = _make_cache_key(chunk, query, weights, provider, model_version,
                          relevance_cutoff, thresholds)

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
    query: Optional[str],
    weights: Optional[dict],
    result: Dict[str, Any],
    provider: Optional[str] = None,
    model_version: Optional[str] = None,
    relevance_cutoff: Optional[float] = None,
    thresholds: Optional[list] = None,
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

    key = _make_cache_key(chunk, query, weights, provider, model_version,
                          relevance_cutoff, thresholds)
    # deepcopy before acquiring the lock so we don't hold it during allocation
    stored = (copy.deepcopy(result), time.time())

    with _cache_lock:
        if len(_cache) >= _CACHE_MAX_SIZE and key not in _cache:
            _cache.popitem(last=False)  # evict least recently used — O(1)
            _cache_stats["evictions"] += 1
        _cache[key] = stored
        _cache.move_to_end(key)  # newest entry is most recently used


def clear_cache(reset_stats: bool = False):
    """
    Clear all cached scores.
    
    Args:
        reset_stats: If True, also reset cache statistics counters.
                     By default, stats remain cumulative across clears.
    """
    with _cache_lock:
        _cache.clear()
        if reset_stats:
            _cache_stats["hits"] = 0
            _cache_stats["misses"] = 0
            _cache_stats["evictions"] = 0


def get_cache_stats() -> Dict[str, Any]:
    """
    Get cache statistics.
    
    Returns:
        Dictionary with hits, misses, evictions, size, and hit_rate
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
    """Reset cache statistics counters to zero."""
    with _cache_lock:
        _cache_stats["hits"] = 0
        _cache_stats["misses"] = 0
        _cache_stats["evictions"] = 0


def print_cache_stats():
    """Print a formatted cache statistics report."""
    stats = get_cache_stats()
    
    print(f"\n{'═' * 60}")
    print(f"  PYMRSF CACHE STATISTICS")
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
    """
    Clear the embedding cache.
    
    Args:
        reset_stats: If True, also reset embedding cache statistics.
                     By default, stats remain cumulative.
    """
    with _embedding_lock:
        _embedding_cache.clear()
        if reset_stats:
            _embedding_stats["hits"] = 0
            _embedding_stats["misses"] = 0
            _embedding_stats["evictions"] = 0
            _embedding_stats["clears"] = 0


def get_embedding_cache_stats() -> Dict[str, Any]:
    """
    Get embedding cache statistics.
    
    Returns:
        Dictionary with hits, misses, clears, size, and hit_rate
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
