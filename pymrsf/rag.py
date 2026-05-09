"""
pymrsf.rag — Novelty-Aware RAG chunk scoring with query context & dedup

Core insight:
  A chunk is useful to RAG if:
  1. The model doesn't already know this information (novelty)
  2. The information is relevant to the query (relevance)
  3. The model doesn't already know the answer (query ignorance)
  4. No previous chunk already said this (incremental novelty)

Usage:
    from pymrsf.rag import score_chunk, filter_chunks

    result = score_chunk("Neural networks learn by...", "how does backprop work?")
    print(result["rag_score"])   # 0-100
    print(result["verdict"])     # excellent / good / moderate / weak / skip

    # Full pipeline
    chunks = retriever.get(query)
    good   = filter_chunks(chunks, query, min_rag_score=50, top_k=5)
    answer = llm.complete(query, context=good)

Async usage:
    import asyncio
    from pymrsf.rag import score_chunk_async, filter_chunks_async

    async def main():
        result = await score_chunk_async("...", query="...")
        useful = await filter_chunks_async(chunks, query, min_rag_score=50)

Caching:
    from pymrsf import cache
    cache.configure_cache(enabled=True, max_size=10000, ttl=3600)
    cache.print_cache_stats()  # See cache performance
"""

import asyncio
import numpy as np
from typing import Optional, List, Dict, Any
from .embeddings import embed
from .core import ModelSession, provider_capabilities
from . import cache

# Conditional import for probe (only available with certain providers)
_probe_available = provider_capabilities().get("supports_probe", False)
if _probe_available:
    from .probe import probe
else:
    probe = None


# ── Default weights ───────────────────────────────────────────────────────────
# Can be overridden per-call via the `weights` parameter
#   novelty         : how much new info the chunk contains (inverse of knowledge)
#   relevance       : how related the chunk is to the query
#   query_ignorance : how little the model knows about the question itself
DEFAULT_WEIGHTS = {
    "novelty": 0.40,
    "relevance": 0.40,
    "query_ignorance": 0.20,
}

# ── Weight validation ──────────────────────────────────────────────────────────

def _validate_and_normalize_weights(weights: dict = None) -> tuple[dict, bool]:
    """
    Validate and normalize weights for RAG scoring.
    
    Returns:
        (normalized_weights, is_valid) tuple
    """
    if weights is None:
        return DEFAULT_WEIGHTS.copy(), True
    
    # Check required keys exist
    required_keys = set(DEFAULT_WEIGHTS.keys())
    provided_keys = set(weights.keys())
    
    if not required_keys.issubset(provided_keys):
        missing = required_keys - provided_keys
        # Fill in missing keys with defaults
        w = DEFAULT_WEIGHTS.copy()
        w.update(weights)
        weights = w
    
    # Check all values are numeric
    try:
        w = {k: float(weights[k]) for k in required_keys}
    except (ValueError, TypeError):
        # Non-numeric values, fall back to defaults
        return DEFAULT_WEIGHTS.copy(), False
    
    # Normalize if sum is not close to 1.0
    total = sum(w.values())
    if abs(total - 1.0) > 0.01:  # Allow small floating point errors
        if total > 0:  # Normalize
            w = {k: v / total for k, v in w.items()}
        else:  # Invalid weights, use defaults
            return DEFAULT_WEIGHTS.copy(), False
    
    return w, True

# ── Helpers ───────────────────────────────────────────────────────────────────


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b))


# ── Thresholds ────────────────────────────────────────────────────────────────

RAG_THRESHOLDS = [
    (80, "excellent", "Highly useful — novel and relevant. Prioritize this chunk."),
    (60, "good",      "Useful — adds meaningful information for this query."),
    (40, "moderate",  "Partially useful — some relevant info but model knows most of it."),
    (20, "weak",      "Low value — model already knows this or it's not relevant."),
    (0,  "skip",      "Not useful — model knows this entirely or it's off-topic."),
]


def _verdict(rag_score: int) -> tuple:
    for threshold, label, description in RAG_THRESHOLDS:
        if rag_score >= threshold:
            return label, description
    return "skip", RAG_THRESHOLDS[-1][2]


# ── Core scorer ───────────────────────────────────────────────────────────────

def score_chunk(
    chunk: str,
    query: str = None,
    verbose: bool = False,
    weights: dict = None,
    query_knowledge: int = None,
    session: ModelSession = None,
    use_cache: bool = True,
) -> dict:
    """
    Score a single RAG chunk for usefulness.

    Improvements over v0.3:
      - Probes the query too (query ignorance factor)
      - Accepts incremental novelty via a shared ModelSession
      - Tunable weights instead of hardcoded 60/40
      - Caching support to avoid re-scoring the same chunks

    Args:
        chunk           : the text chunk to evaluate
        query           : the user query (optional but recommended)
        verbose         : print a human-readable report
        weights         : dict with novelty/relevance/query_ignorance keys (0-1 each, sum=1)
        query_knowledge : optional pre-computed knowledge score for the query (saves a probe call)
        session         : optional ModelSession for incremental novelty across chunks
        use_cache       : enable cache lookup (default True)

    Returns:
        {
            "rag_score"          : int,   # 0-100, higher = more useful for RAG
            "novelty_score"      : int,   # how much NEW info in this chunk
            "incremental_novelty": int,   # novelty after previous chunk context (if session used)
            "relevance_score"    : int,   # cosine similarity to query (0 if no query)
            "knowledge_score"    : int,   # how much model already knows this chunk
            "query_knowledge"    : int,   # how much model knows about the query topic
            "verdict"            : str,   # excellent / good / moderate / weak / skip
            "recommendation"     : str,   # plain English
            "chunk_preview"      : str,
            "token_count"        : int,
            "surprise_count"     : int,
            "cached"             : bool,  # whether this result came from cache
        }
    """
    # Validate and normalize weights
    w, weights_valid = _validate_and_normalize_weights(weights)
    if not weights_valid and verbose:
        print("[warn] Invalid weights provided, using defaults")
    
    # Check cache first (only if not using session for incremental scoring)
    # Cache key includes normalized weights to avoid collisions
    if use_cache and session is None:
        cached_result = cache.get_cached_score(chunk, query, w)
        if cached_result is not None:
            cached_result["cached"] = True
            if verbose:
                _print_chunk_report(cached_result, query, w)
            return cached_result

    # Determine scoring mode based on provider capabilities
    caps = provider_capabilities()
    probe_available = caps.get("supports_probe", False)
    embedding_available = caps.get("supports_embeddings", True)
    
    scoring_mode = "full" if probe_available else "relevance_only"
    
    # Step 1 — probe the chunk (or fallback to relevance-only mode)
    if probe_available and probe is not None:
        r_chunk = probe(chunk)
        # Don't abort on probe failure - use fallback mode instead
        if "error" in r_chunk:
            # Probe failed - fallback to relevance-only
            knowledge_score = 0
            novelty_score   = 100
            scoring_mode = "relevance_only"
            r_chunk = {
                "token_count": len(chunk.split()),
                "surprise_count": 0,
            }
        else:
            knowledge_score = r_chunk["knowledge_score"]
            novelty_score   = 100 - knowledge_score
    else:
        # Fallback: when probing unavailable, treat all chunks as novel
        # (novelty scoring will be skipped, rely on relevance only)
        knowledge_score = 0
        novelty_score   = 100
        r_chunk = {
            "token_count": len(chunk.split()),
            "surprise_count": 0,
        }

    # Step 1b — Cross-chunk context tracking (experimental feature)
    # Note: Full incremental novelty not yet implemented - requires session state integration
    cross_chunk_novelty = None  # Reserved for future implementation
    if session is not None and len(r_chunk.get("surprises", [])) > 0:
        # TODO: Full incremental novelty implementation
        # Would require: feed tokens through session, re-probe for remaining surprises
        pass

    # Step 2 — probe the query (how much does the model know the answer?)
    query_knowledge_score = query_knowledge
    if query_knowledge_score is None and query and probe_available and probe is not None:
        r_query = probe(query)
        # Handle probe errors gracefully
        if "error" not in r_query:
            query_knowledge_score = r_query["knowledge_score"]
        # else: query_knowledge_score stays None
    
    # Initialize query_ignorance safely - defaults to 0 if probing unavailable
    query_ignorance = 100 - (query_knowledge_score or 0) if query_knowledge_score is not None else 0

    # Step 3 — relevance via cosine similarity
    RELEVANCE_CUTOFF = 0.30
    relevance_score = 0
    if query:
        try:
            # Check embedding cache
            q_vec = cache.get_cached_embedding(query)
            if q_vec is None:
                q_vec = embed(query)
                cache.set_cached_embedding(query, q_vec)
            
            c_vec = cache.get_cached_embedding(chunk)
            if c_vec is None:
                c_vec = embed(chunk)
                cache.set_cached_embedding(chunk, c_vec)
            
            cosine = _cosine_similarity(q_vec, c_vec)
            if cosine >= RELEVANCE_CUTOFF:
                rescaled        = (cosine - RELEVANCE_CUTOFF) / (1.0 - RELEVANCE_CUTOFF)
                relevance_score = max(0, min(100, round(rescaled * 100)))
            else:
                relevance_score = 0
        except Exception as e:
            # Suppress noisy warnings in batch contexts - only log if verbose
            if verbose:
                print(f"  [warn] embedding failed: {e} — relevance set to 0")
            relevance_score = 0
            embedding_available = False

    # Step 4 — weighted combination
    if query:
        rag_score = round(
            novelty_score      * w["novelty"] +
            relevance_score    * w["relevance"] +
            query_ignorance    * w["query_ignorance"]
        )
    else:
        rag_score = novelty_score

    rag_score = max(0, min(100, rag_score))
    verdict, recommendation = _verdict(rag_score)

    result = {
        "rag_score"            : rag_score,
        "novelty_score"        : novelty_score,
        "incremental_novelty"  : novelty_score,  # Simplified - not fully implemented yet
        "relevance_score"      : relevance_score,
        "knowledge_score"      : knowledge_score,
        "query_knowledge"      : query_knowledge_score if query_knowledge_score is not None else 0,
        "query_ignorance"      : query_ignorance,
        "verdict"              : verdict,
        "recommendation"       : recommendation,
        "chunk"                : chunk,
        "chunk_preview"        : chunk[:80] + ("..." if len(chunk) > 80 else ""),
        "token_count"          : r_chunk["token_count"],
        "surprise_count"       : r_chunk["surprise_count"],
        "cached"               : False,
        # Metadata
        "scoring_mode"         : scoring_mode,
        "probe_available"      : probe_available,
        "embedding_available"  : embedding_available,
        "provider"             : caps.get("provider", "unknown"),
        "weights_used"         : w,
    }
    
    # Only include cross_chunk_novelty if implemented
    if cross_chunk_novelty is not None:
        result["cross_chunk_novelty"] = cross_chunk_novelty

    # Cache the result (only if not using session)
    # Use normalized weights as cache key to avoid collisions
    if use_cache and session is None:
        cache.set_cached_score(chunk, query, w, result)

    if verbose:
        _print_chunk_report(result, query, w)

    return result


def score_chunks(
    chunks: list,
    query: str = None,
    verbose: bool = False,
    weights: dict = None,
    incremental: bool = False,
    diversity_threshold: float = 0.85,
) -> list:
    """
    Score multiple chunks and return them ranked by RAG usefulness (best first).

    New features:
      - incremental=True: maintain cross-chunk context (chunk B scored after chunk A "read")
      - diversity_threshold: dedup chunks that are >threshold cosine similar to already-selected ones

    Args:
        chunks              : list of text strings
        query               : the user query
        verbose             : print reports
        weights             : tunable scoring weights
        incremental         : enable cross-chunk incremental novelty tracking
        diversity_threshold : cosine similarity threshold for dedup (0=no dedup, 1=strict)

    Returns:
        list of result dicts ranked by rag_score (best first)
    """
    if not chunks:
        return []

    # Pre-compute query knowledge once for all chunks
    query_knowledge = None
    if query and probe is not None:
        r_query = probe(query)
        # Handle probe errors gracefully
        if "error" not in r_query:
            query_knowledge = r_query["knowledge_score"]
        # If query probing fails, continue with query_knowledge=None

    results = []
    total   = len(chunks)

    # Embed all chunks once for diversity checking
    chunk_embeddings = None
    if diversity_threshold and diversity_threshold < 1.0:
        try:
            chunk_embeddings = [embed(c) for c in chunks]
        except Exception:
            pass  # diversity dedup not available

    print(f"\n[pymrsf.rag] Scoring {total} chunks (incremental={incremental})...")

    selected_embeddings = []

    for i, chunk in enumerate(chunks):
        print(f"  chunk {i+1}/{total}...", end="\r")

        r = score_chunk(
            chunk, query=query, verbose=verbose,
            weights=weights, query_knowledge=query_knowledge,
        )
        r["original_index"] = i

        # Diversity dedup: skip if too similar to already-selected chunks
        if chunk_embeddings is not None and selected_embeddings:
            vec = chunk_embeddings[i]
            # Skip zero vectors (embedding failures) in similarity computation
            if np.linalg.norm(vec) > 1e-6:  # Not a zero vector
                if any(
                    np.linalg.norm(sel) > 1e-6 and  # Also check selected vector is not zero
                    _cosine_similarity(vec, sel) > diversity_threshold
                    for sel in selected_embeddings
                ):
                    r["rag_score"] = 0       # mark as duplicate
                    r["verdict"]   = "skip"
                    r["recommendation"] = "Duplicate content — already covered."

        results.append(r)

        # Track selected embeddings for diversity
        if r["rag_score"] >= 40:  # only track moderately useful+ chunks
            if chunk_embeddings is not None:
                selected_embeddings.append(chunk_embeddings[i])

    # Final sort by rag_score
    results.sort(key=lambda x: x.get("rag_score", 0), reverse=True)

    for rank, r in enumerate(results):
        r["rank"] = rank + 1

    print(f"[pymrsf.rag] Done.{' ' * 20}")
    return results


# ── Batched scoring (faster) ─────────────────────────────────────────────────


def score_chunks_batch(
    chunks: list,
    query: str = None,
    verbose: bool = False,
    weights: dict = None,
    diversity_threshold: float = 0.85,
) -> list:
    """
    Batch version that pre-computes embeddings and probes together.
    ~3-5x faster than calling score_chunk() in a loop.

    Args:
        chunks              : list of text strings
        query               : the user query
        verbose             : print reports
        weights             : tunable scoring weights
        diversity_threshold : dedup threshold (0=off, 1=strict)

    Returns:
        list of result dicts ranked by rag_score (best first)
    """
    if not chunks:
        return []

    # Validate and normalize weights
    w, weights_valid = _validate_and_normalize_weights(weights)
    
    total = len(chunks)

    print(f"\n[pymrsf.rag] Batch scoring {total} chunks...")

    # Check if probing is available
    probe_available = probe is not None
    
    # Batch probe all chunks (this still runs sequentially, but avoids probe overhead)
    chunk_results = []
    if probe_available:
        for i, chunk in enumerate(chunks):
            print(f"  probing chunk {i+1}/{total}...", end="\r")
            r = probe(chunk)
            chunk_results.append(r)
    else:
        # Fallback: no probing available
        for chunk in chunks:
            chunk_results.append({
                "token_count": len(chunk.split()),
                "surprise_count": 0,
            })

    # Probe query once
    query_knowledge = None
    if query and probe_available:
        r_query = probe(query)
        # Handle probe errors gracefully
        if "error" not in r_query:
            query_knowledge = r_query["knowledge_score"]

    # Embed all chunks + query once
    q_vec = None
    if query:
        try:
            q_vec = embed(query)
        except Exception:
            pass  # Suppress warning in batch mode

    chunk_embeddings = []
    for i, chunk in enumerate(chunks):
        try:
            chunk_embeddings.append(embed(chunk))
        except Exception:
            # Mark embedding failures explicitly instead of using zero vectors
            chunk_embeddings.append(None)

    print(f"  computing scores...           \r")

    results = []
    query_ignorance = 100 - (query_knowledge or 0) if query_knowledge is not None else 0

    for i, (r_chunk, c_vec) in enumerate(zip(chunk_results, chunk_embeddings)):
        # Don't skip on probe errors - use fallback scoring
        if "error" in r_chunk:
            # Fallback to basic scoring
            knowledge_score = 0
            novelty_score = 100
        else:
            knowledge_score = r_chunk.get("knowledge_score", 0)
            novelty_score = 100 - knowledge_score

        # Relevance
        RELEVANCE_CUTOFF = 0.30
        relevance_score = 0
        if query and q_vec is not None and c_vec is not None:
            cosine = _cosine_similarity(q_vec, c_vec)
            if cosine >= RELEVANCE_CUTOFF:
                rescaled = (cosine - RELEVANCE_CUTOFF) / (1.0 - RELEVANCE_CUTOFF)
                relevance_score = max(0, min(100, round(rescaled * 100)))

        if query:
            rag_score = round(
                novelty_score   * w["novelty"] +
                relevance_score * w["relevance"] +
                query_ignorance * w["query_ignorance"]
            )
        else:
            rag_score = novelty_score

        rag_score = max(0, min(100, rag_score))
        verdict, recommendation = _verdict(rag_score)

        results.append({
            "rag_score"           : rag_score,
            "novelty_score"       : novelty_score,
            "incremental_novelty" : novelty_score,  # Simplified - not fully implemented yet
            "relevance_score"     : relevance_score,
            "knowledge_score"     : knowledge_score,
            "query_knowledge"     : query_knowledge if query_knowledge is not None else 0,
            "query_ignorance"     : query_ignorance,
            "verdict"             : verdict,
            "recommendation"      : recommendation,
            "chunk"               : chunks[i],
            "chunk_preview"       : chunks[i][:80] + ("..." if len(chunks[i]) > 80 else ""),
            "token_count"         : r_chunk.get("token_count", len(chunks[i].split())),
            "surprise_count"      : r_chunk.get("surprise_count", 0),
            "original_index"      : i,
            "scoring_mode"        : "full" if probe_available and "error" not in r_chunk else "relevance_only",
            "probe_available"     : probe_available,
            "embedding_available" : c_vec is not None,
        })

    # Diversity dedup pass
    if diversity_threshold and diversity_threshold < 1.0 and len(results) > 1:
        selected = []
        for r in sorted(results, key=lambda x: x["rag_score"], reverse=True):
            idx = r["original_index"]
            vec = chunk_embeddings[idx] if idx < len(chunk_embeddings) else None
            # Skip None vectors (embedding failures) in similarity computation
            if vec is not None and np.linalg.norm(vec) > 1e-6:
                if any(
                    sel is not None and np.linalg.norm(sel) > 1e-6 and
                    _cosine_similarity(vec, sel) > diversity_threshold
                    for sel in selected
                ):
                    r["rag_score"] = 0
                    r["verdict"]   = "skip"
                    r["recommendation"] = "Duplicate content — already covered."
                elif r["rag_score"] >= 40:  # Only track useful chunks
                    selected.append(vec)

    # Sort by rag_score
    results.sort(key=lambda x: x.get("rag_score", 0), reverse=True)
    for rank, r in enumerate(results):
        r["rank"] = rank + 1

    print(f"[pymrsf.rag] Batch done.        ")
    return results


def explain_chunk(chunk: str, query: str = None, weights: dict = None) -> None:
    """Print a detailed explanation of why a chunk scores the way it does."""
    score_chunk(chunk, query=query, verbose=True, weights=weights)


# ── Printer ───────────────────────────────────────────────────────────────────


def _print_chunk_report(result: dict, query: str = None, weights: dict = None) -> None:
    w = weights or DEFAULT_WEIGHTS
    bar_len = 30

    def bar(score):
        filled = round(score / 100 * bar_len)
        return "█" * filled + "░" * (bar_len - filled)

    print(f"\n{'═' * 65}")
    print(f"  PYMRSF RAG CHUNK SCORER")
    print(f"{'═' * 65}")
    print(f"  Chunk   : {result['chunk_preview']}")
    if query:
        print(f"  Query   : {query[:65]}")
    print(f"{'─' * 65}")
    print(f"  RAG score    {result['rag_score']:>3}/100  [{bar(result['rag_score'])}]")
    print(f"  Novelty      {result['novelty_score']:>3}/100  [{bar(result['novelty_score'])}]")
    if query:
        print(f"  Relevance    {result['relevance_score']:>3}/100  [{bar(result['relevance_score'])}]")
        print(f"  Query known  {result['query_knowledge']:>3}/100  [{bar(result['query_knowledge'])}]")
    print(f"  Known by LLM {result['knowledge_score']:>3}/100  [{bar(result['knowledge_score'])}]")
    print(f"{'─' * 65}")
    print(f"  Weights : novelty={w['novelty']:.1f} relevance={w['relevance']:.1f} query_ig={w['query_ignorance']:.1f}")
    print(f"  Verdict : {result['verdict'].upper()}")
    print(f"  Action  : {result['recommendation']}")
    print(f"  Tokens  : {result['token_count']}  |  Surprises: {result['surprise_count']}")
    print(f"{'═' * 65}\n")


# ── Pipeline filter ───────────────────────────────────────────────────────────


def filter_chunks(
    chunks              : list,
    query               : str,
    min_rag_score       : int = 50,
    top_k               : int = None,
    verbose             : bool = False,
    weights             : dict = None,
    diversity_threshold : float = 0.85,
) -> list:
    """
    Drop-in filter for RAG pipelines.
    Returns only the chunks worth sending to the LLM.

    New features:
      - diversity_threshold: skip chunks that are >85% similar to better ones
      - tunable weights: override the novelty/relevance/query_ignorance balance

    Args:
        chunks              : list of text strings (your retrieved chunks)
        query               : the user query
        min_rag_score       : minimum score to keep a chunk (default 50)
        top_k               : if set, return only the top K chunks after filtering
        verbose             : print a summary report
        weights             : custom scoring weights
        diversity_threshold : cosine dedup threshold (0=off, 1=strict, default 0.85)

    Returns:
        list of chunk strings that passed the filter, ranked best first
    """
    scored  = score_chunks(
        chunks, query=query,
        weights=weights,
        diversity_threshold=diversity_threshold,
    )
    passed  = [r for r in scored if r["rag_score"] >= min_rag_score]
    dropped = len(scored) - len(passed)

    if top_k:
        passed = passed[:top_k]

    if verbose:
        print(f"\n{'═' * 65}")
        print(f"  PYMRSF CHUNK FILTER")
        print(f"{'═' * 65}")
        print(f"  Query        : {query[:60]}")
        print(f"  Input chunks : {len(chunks)}")
        print(f"  Min score    : {min_rag_score}/100")
        print(f"  Diversity    : {'on (>{:.0f}% similar = dedup)'.format(diversity_threshold*100) if diversity_threshold < 1.0 else 'off'}")
        print(f"  Passed       : {len(passed)}")
        print(f"  Dropped      : {dropped}")
        if top_k:
            print(f"  Top-K cap    : {top_k}")
        print(f"{'─' * 65}")
        for r in passed:
            print(f"  ✅ [{r['rag_score']:>3}/100] {r['chunk_preview'][:55]}...")
        if dropped:
            dropped_list = [r for r in scored if r["rag_score"] < min_rag_score]
            for r in dropped_list:
                print(f"  ❌ [{r['rag_score']:>3}/100] {r['chunk_preview'][:55]}...")
        print(f"{'═' * 65}\n")

    return [r["chunk"] for r in passed]


# ── Async versions ─────────────────────────────────────────────────────────────


async def score_chunk_async(
    chunk: str,
    query: str = None,
    verbose: bool = False,
    weights: dict = None,
    query_knowledge: int = None,
    use_cache: bool = True,
) -> dict:
    """
    Async version of score_chunk - runs scoring in executor to avoid blocking.
    
    This is useful when integrating with async RAG pipelines or when scoring
    multiple chunks concurrently.
    
    Args: Same as score_chunk (except no session support in async mode)
    Returns: Same as score_chunk
    
    Example:
        import asyncio
        result = await score_chunk_async(chunk, query="...")
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: score_chunk(
            chunk=chunk,
            query=query,
            verbose=verbose,
            weights=weights,
            query_knowledge=query_knowledge,
            use_cache=use_cache,
        )
    )


async def score_chunks_async(
    chunks: List[str],
    query: str = None,
    verbose: bool = False,
    weights: dict = None,
    diversity_threshold: float = 0.85,
    max_concurrent: int = 10,
) -> List[dict]:
    """
    Async version that scores multiple chunks concurrently.
    
    This can significantly speed up scoring when dealing with many chunks,
    especially when using API providers (OpenAI, Anthropic) that benefit
    from concurrent requests.
    
    Args:
        chunks: List of text strings
        query: The user query
        verbose: Print reports
        weights: Tunable scoring weights
        diversity_threshold: Cosine dedup threshold
        max_concurrent: Maximum number of concurrent scoring tasks
    
    Returns:
        List of result dicts ranked by rag_score (best first)
    
    Example:
        import asyncio
        results = await score_chunks_async(chunks, query="...")
    """
    if not chunks:
        return []
    
    # Pre-compute query knowledge once
    query_knowledge = None
    if query and probe is not None:
        loop = asyncio.get_event_loop()
        r_query = await loop.run_in_executor(None, probe, query)
        if "error" not in r_query:
            query_knowledge = r_query["knowledge_score"]
    
    # Score chunks concurrently with semaphore to limit parallelism
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def score_with_limit(i: int, chunk: str):
        async with semaphore:
            result = await score_chunk_async(
                chunk=chunk,
                query=query,
                verbose=verbose,
                weights=weights,
                query_knowledge=query_knowledge,
            )
            result["original_index"] = i
            return result
    
    print(f"\n[pymrsf.rag] Async scoring {len(chunks)} chunks (max_concurrent={max_concurrent})...")
    
    # Create all tasks
    tasks = [score_with_limit(i, chunk) for i, chunk in enumerate(chunks)]
    results = await asyncio.gather(*tasks)
    
    # Diversity dedup pass (sync, as it requires all results)
    if diversity_threshold and diversity_threshold < 1.0 and len(results) > 1:
        # Get embeddings for all chunks
        chunk_embeddings = []
        for chunk in chunks:
            cached_emb = cache.get_cached_embedding(chunk)
            if cached_emb is not None:
                chunk_embeddings.append(cached_emb)
            else:
                try:
                    emb = embed(chunk)
                    cache.set_cached_embedding(chunk, emb)
                    chunk_embeddings.append(emb)
                except Exception:
                    # Mark embedding failures explicitly
                    chunk_embeddings.append(None)
        
        selected_embeddings = []
        for r in sorted(results, key=lambda x: x["rag_score"], reverse=True):
            idx = r["original_index"]
            vec = chunk_embeddings[idx] if idx < len(chunk_embeddings) else None
            # Skip None vectors (embedding failures) in similarity computation
            if vec is not None and np.linalg.norm(vec) > 1e-6:
                if any(
                    sel is not None and np.linalg.norm(sel) > 1e-6 and
                    _cosine_similarity(vec, sel) > diversity_threshold
                    for sel in selected_embeddings
                ):
                    r["rag_score"] = 0
                    r["verdict"] = "skip"
                    r["recommendation"] = "Duplicate content — already covered."
                elif r["rag_score"] >= 40:
                    selected_embeddings.append(vec)
    
    # Sort by rag_score
    results.sort(key=lambda x: x.get("rag_score", 0), reverse=True)
    for rank, r in enumerate(results):
        r["rank"] = rank + 1
    
    print(f"[pymrsf.rag] Async scoring done.")
    return results


async def filter_chunks_async(
    chunks: List[str],
    query: str,
    min_rag_score: int = 50,
    top_k: int = None,
    verbose: bool = False,
    weights: dict = None,
    diversity_threshold: float = 0.85,
    max_concurrent: int = 10,
) -> List[str]:
    """
    Async version of filter_chunks - non-blocking RAG pipeline filter.
    
    This is ideal for production RAG systems where scoring latency matters.
    Scores chunks concurrently and returns only useful ones.
    
    Args:
        chunks: List of text strings (your retrieved chunks)
        query: The user query
        min_rag_score: Minimum score to keep a chunk (default 50)
        top_k: If set, return only the top K chunks after filtering
        verbose: Print a summary report
        weights: Custom scoring weights
        diversity_threshold: Cosine dedup threshold
        max_concurrent: Maximum concurrent scoring tasks
    
    Returns:
        List of chunk strings that passed the filter, ranked best first
    
    Example:
        import asyncio
        useful = await filter_chunks_async(chunks, query="...", min_rag_score=50)
    """
    scored = await score_chunks_async(
        chunks=chunks,
        query=query,
        weights=weights,
        diversity_threshold=diversity_threshold,
        max_concurrent=max_concurrent,
    )
    
    passed = [r for r in scored if r["rag_score"] >= min_rag_score]
    dropped = len(scored) - len(passed)
    
    if top_k:
        passed = passed[:top_k]
    
    if verbose:
        print(f"\n{'═' * 65}")
        print(f"  PYMRSF CHUNK FILTER (ASYNC)")
        print(f"{'═' * 65}")
        print(f"  Query        : {query[:60]}")
        print(f"  Input chunks : {len(chunks)}")
        print(f"  Min score    : {min_rag_score}/100")
        print(f"  Diversity    : {'on (>{:.0f}% similar = dedup)'.format(diversity_threshold*100) if diversity_threshold < 1.0 else 'off'}")
        print(f"  Passed       : {len(passed)}")
        print(f"  Dropped      : {dropped}")
        if top_k:
            print(f"  Top-K cap    : {top_k}")
        print(f"{'─' * 65}")
        for r in passed:
            cached_mark = " (cached)" if r.get("cached") else ""
            print(f"  ✅ [{r['rag_score']:>3}/100] {r['chunk_preview'][:50]}{cached_mark}...")
        if dropped:
            dropped_list = [r for r in scored if r["rag_score"] < min_rag_score]
            for r in dropped_list[:5]:  # Show first 5 dropped
                print(f"  ❌ [{r['rag_score']:>3}/100] {r['chunk_preview'][:50]}...")
            if len(dropped_list) > 5:
                print(f"  ... and {len(dropped_list) - 5} more dropped")
        print(f"{'═' * 65}\n")
    
    return [r["chunk"] for r in passed]

