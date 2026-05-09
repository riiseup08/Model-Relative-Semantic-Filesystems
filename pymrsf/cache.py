"""
pymrsf.cache — Caching layer for RAG chunk scores

Avoids re-scoring the same chunks across multiple queries.
Thread-safe, configurable cache with statistics.

Caching Strategy:
  - Score cache: Time-based eviction (removes oldest entry by timestamp)
  - Embedding cache: Wholesale clear when limit reached
  
Note: This is not a strict access-order LRU cache. The score cache evicts
based on insertion time, not access time.
"""

import hashlib
import time
import copy
from typing import Optional, Dict, Any
from functools import lru_cache
import threading


# ── Cache configuration ────────────────────────────────────────────────────────

_CACHE_ENABLED = True
_CACHE_MAX_SIZE = 10000  # Maximum number of cached scores
_CACHE_TTL = 3600  # Time-to-live in seconds (1 hour default)
_EMBEDDING_CACHE_MAX_SIZE = 5000  # Separate limit for embeddings
_EMBEDDING_CACHE_TTL = 7200  # Longer TTL for embeddings (2 hours)

# Global cache and statistics
_cache: Dict[str, tuple] = {}  # key -> (result, timestamp)
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
    model_version: Optional[str] = None
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
    query: Optional[str] = None, 
    weights: Optional[dict] = None,
    provider: Optional[str] = None,
    model_version: Optional[str] = None
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
    
    key = _make_cache_key(chunk, query, weights, provider, model_version)
    
    with _cache_lock:
        if key in _cache:
            result, timestamp = _cache[key]
            
            # Check TTL
            if _CACHE_TTL > 0 and (time.time() - timestamp) > _CACHE_TTL:
                # Expired - remove it
                del _cache[key]
                _cache_stats["misses"] += 1
                return None
            
            _cache_stats["hits"] += 1
            # Deep copy to prevent mutations from affecting cached data
            return copy.deepcopy(result)
        
        _cache_stats["misses"] += 1
        return None


def set_cached_score(
    chunk: str, 
    query: Optional[str], 
    weights: Optional[dict], 
    result: Dict[str, Any],
    provider: Optional[str] = None,
    model_version: Optional[str] = None
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
    
    with _cache_lock:
        # Time-based eviction if cache is full (remove oldest by timestamp)
        if len(_cache) >= _CACHE_MAX_SIZE:
            # Evict oldest entry by timestamp
            oldest_key = min(_cache.keys(), key=lambda k: _cache[k][1])
            del _cache[oldest_key]
            _cache_stats["evictions"] += 1
        
        # Deep copy to prevent external mutations from affecting cache
        _cache[key] = (copy.deepcopy(result), time.time())


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

@lru_cache(maxsize=1000)
def _cached_text_hash(text: str) -> str:
    """Cache text hashes for embedding lookups."""
    return hashlib.sha256(text.encode()).hexdigest()


_embedding_cache: Dict[str, tuple] = {}  # hash -> (embedding, timestamp)
_embedding_lock = threading.Lock()
_embedding_stats = {
    "hits": 0,
    "misses": 0,
    "clears": 0,  # Counts wholesale clear events
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
            
            # Check TTL for embeddings
            if _EMBEDDING_CACHE_TTL > 0 and (time.time() - timestamp) > _EMBEDDING_CACHE_TTL:
                # Expired - remove it
                del _embedding_cache[text_hash]
                _embedding_stats["misses"] += 1
                return None
            
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
        # Wholesale clear when limit reached (not true LRU)
        if len(_embedding_cache) >= _EMBEDDING_CACHE_MAX_SIZE:
            _embedding_cache.clear()
            _embedding_stats["clears"] += 1
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
            "clears": _embedding_stats["clears"],
            "size": len(_embedding_cache),
            "max_size": _EMBEDDING_CACHE_MAX_SIZE,
            "hit_rate": round(hit_rate * 100, 2),
            "enabled": _CACHE_ENABLED,
        }
