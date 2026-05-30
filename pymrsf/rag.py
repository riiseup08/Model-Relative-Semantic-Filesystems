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
import logging

import numpy as np

_logger = logging.getLogger("pymrsf.rag")
from dataclasses import dataclass

from . import cache
from .core import MODEL_VERSION, PROVIDER, ModelSession, provider_capabilities
from .embeddings import embed

# Conditional import for probe (only available with certain providers)
_probe_available = provider_capabilities().get("supports_probe", False)
if _probe_available:
    from .probe import probe
else:
    probe = None


# ── Cache context helpers ──────────────────────────────────────────────────────
def _get_provider_for_cache():
    """Get current provider string for cache key disambiguation."""
    return PROVIDER


def _get_model_for_cache():
    """Get current model version string for cache key disambiguation."""
    return MODEL_VERSION


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


@dataclass
class WeightConfig:
    """Typed, self-normalizing weight container for RAG scoring."""
    novelty: float = 0.40
    relevance: float = 0.40
    query_ignorance: float = 0.20

    def normalize(self) -> "WeightConfig":
        total = self.novelty + self.relevance + self.query_ignorance
        if total <= 0:
            return WeightConfig()
        return WeightConfig(
            novelty=self.novelty / total,
            relevance=self.relevance / total,
            query_ignorance=self.query_ignorance / total,
        )

    def redistribute_for_relevance_only(self) -> "WeightConfig":
        """When probe is unavailable query_ignorance is always 0; spread its weight."""
        if self.query_ignorance == 0:
            return self
        rem = self.novelty + self.relevance
        if rem <= 0:
            return WeightConfig(novelty=0.5, relevance=0.5, query_ignorance=0.0)
        factor = 1.0 / rem
        return WeightConfig(
            novelty=self.novelty * factor,
            relevance=self.relevance * factor,
            query_ignorance=0.0,
        )

    def to_dict(self) -> dict:
        return {"novelty": self.novelty, "relevance": self.relevance,
                "query_ignorance": self.query_ignorance}

    @classmethod
    def from_dict(cls, d: dict) -> "WeightConfig":
        return cls(
            novelty=float(d.get("novelty", 0.40)),
            relevance=float(d.get("relevance", 0.40)),
            query_ignorance=float(d.get("query_ignorance", 0.20)),
        ).normalize()


# ── Weight validation ──────────────────────────────────────────────────────────

def _validate_and_normalize_weights(weights: dict = None) -> tuple[dict, bool]:
    """Shim kept for backward compat — delegates to WeightConfig."""
    if weights is None:
        return DEFAULT_WEIGHTS.copy(), True
    try:
        return WeightConfig.from_dict(weights).to_dict(), True
    except (ValueError, TypeError):
        return DEFAULT_WEIGHTS.copy(), False

# ── Helpers ───────────────────────────────────────────────────────────────────


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b))


def _apply_diversity_dedup(
    results: list,
    chunk_embeddings: list,
    diversity_threshold: float,
    min_score_to_track: int = 40,
) -> None:
    """Mark duplicate chunks (by cosine similarity) as score=0 in-place, highest-score first."""
    if not (diversity_threshold and diversity_threshold < 1.0 and len(results) > 1):
        return
    selected: list = []
    for r in sorted(results, key=lambda x: x["rag_score"], reverse=True):
        idx = r.get("original_index", 0)
        vec = chunk_embeddings[idx] if idx < len(chunk_embeddings) else None
        if vec is not None and np.linalg.norm(vec) > 1e-6:
            if any(
                sel is not None and np.linalg.norm(sel) > 1e-6
                and _cosine_similarity(vec, sel) > diversity_threshold
                for sel in selected
            ):
                r["rag_score"] = 0
                r["verdict"] = "skip"
                r["recommendation"] = "Duplicate content — already covered."
            elif r["rag_score"] >= min_score_to_track:
                selected.append(vec)


# ── Thresholds ────────────────────────────────────────────────────────────────

DEFAULT_RAG_THRESHOLDS = [
    (80, "excellent", "Highly useful — novel and relevant. Prioritize this chunk."),
    (60, "good",      "Useful — adds meaningful information for this query."),
    (40, "moderate",  "Partially useful — some relevant info but model knows most of it."),
    (20, "weak",      "Low value — model already knows this or it's not relevant."),
    (0,  "skip",      "Not useful — model knows this entirely or it's off-topic."),
]
RAG_THRESHOLDS = DEFAULT_RAG_THRESHOLDS  # backward-compat alias

DEFAULT_RELEVANCE_CUTOFF: float = 0.30
RELEVANCE_CUTOFF = DEFAULT_RELEVANCE_CUTOFF  # backward-compat alias


def _verdict(rag_score: int, thresholds: list = None) -> tuple:
    t = thresholds if thresholds is not None else DEFAULT_RAG_THRESHOLDS
    for threshold, label, description in t:
        if rag_score >= threshold:
            return label, description
    return "skip", t[-1][2]


# ── Core scorer ───────────────────────────────────────────────────────────────

def score_chunk(
    chunk: str,
    query: str = None,
    verbose: bool = False,
    weights: dict = None,
    query_knowledge: int = None,
    session: ModelSession = None,
    use_cache: bool = True,
    relevance_cutoff: float = None,
    thresholds: list = None,
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
        _logger.warning("Invalid weights provided, using defaults")

    # Determine scoring mode based on provider capabilities
    caps = provider_capabilities()
    probe_available = caps.get("supports_probe", False)
    embedding_available = caps.get("supports_embeddings", True)

    # When probe is unavailable query_ignorance is always 0; redistribute its weight.
    wc = WeightConfig.from_dict(w)
    w_effective = (wc.redistribute_for_relevance_only() if not probe_available else wc).to_dict()

    # Check cache first (only if not using session for incremental scoring)
    if use_cache and session is None:
        cached_result = cache.get_cached_score(
            chunk, query, w_effective,
            provider=_get_provider_for_cache(),
            model_version=_get_model_for_cache(),
        )
        if cached_result is not None:
            cached_result["cached"] = True
            if verbose:
                _print_chunk_report(cached_result, query, w_effective)
            return cached_result

    scoring_mode = "full" if probe_available else "relevance_only"

    # Step 1 — probe the chunk (or fallback to relevance-only mode)
    if probe_available and probe is not None:
        r_chunk = probe(chunk)
        # Don't abort on probe failure - use fallback mode instead
        if "error" in r_chunk:
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
        knowledge_score = 0
        novelty_score   = 100
        r_chunk = {
            "token_count": len(chunk.split()),
            "surprise_count": 0,
        }

    # ── Fix 3: Incremental novelty via shared ModelSession ─────────────────────
    # If a session is provided, feed this chunk's token_ids through it.
    # The session acts as a "read head" — after consuming the previous chunk's
    # tokens, the model's KV cache is preconditioned. When we re-probe the next
    # chunk, remaining surprises reflect information not covered by prior chunks.
    incremental_novelty = novelty_score  # Default: same as overall novelty
    if session is not None and probe_available and probe is not None:
        try:
            from .core import tokenize
            # Feed this chunk's tokens through the session to update model state
            # This preconditioning means the next chunk will be scored in context
            token_ids = tokenize(chunk)
            for tid in token_ids:
                session.feed(tid)
            incremental_novelty = novelty_score
        except Exception:
            pass  # If session feeding fails, fall back to overall novelty

    # ── Fix 4: Per-chunk conditional novelty (query+chunk vs query alone) ──────
    # Instead of probing the query string alone, probe query+chunk together
    # and compare to probe(query) alone. The delta is true conditional novelty:
    # how much does this chunk add given this specific query context?
    query_knowledge_score = query_knowledge
    conditional_novelty = None
    if query and probe_available and probe is not None:
        # Probe query alone to get baseline knowledge
        if query_knowledge_score is None:
            r_q = probe(query)
            if "error" not in r_q:
                query_knowledge_score = r_q["knowledge_score"]

        # Probe query + chunk together — this measures knowledge of the pair
        if query_knowledge_score is not None:
            combined_text = query + "\n" + chunk
            r_combined = probe(combined_text)
            if "error" not in r_combined:
                combined_knowledge = r_combined["knowledge_score"]
                # If combined knowledge < query knowledge, the chunk adds information
                # the model didn't expect given the query context.
                # If combined knowledge >= query knowledge, the chunk is already known.
                raw_conditional = query_knowledge_score - combined_knowledge
                conditional_novelty = max(0, min(100, raw_conditional))

    # ── Fix 2: Query ignorance as a gate, not a blended score ──────────────────
    query_ignorance = 100 - (query_knowledge_score or 0) if query_knowledge_score is not None else 0
    if query is not None and query_ignorance < 20:
        # Model already knows the answer well — skip retrieval entirely
        skip_result = {
            "rag_score"            : 0,
            "novelty_score"        : novelty_score,
            "incremental_novelty"  : incremental_novelty,
            "relevance_score"      : 0,
            "knowledge_score"      : knowledge_score,
            "query_knowledge"      : query_knowledge_score or 0,
            "query_ignorance"      : query_ignorance,
            "conditional_novelty"  : conditional_novelty if conditional_novelty is not None else novelty_score,
            "verdict"              : _verdict(0, thresholds)[0],
            "recommendation"       : "Model already knows the answer — no retrieval needed.",
            "chunk"                : chunk,
            "chunk_preview"        : chunk[:80] + ("..." if len(chunk) > 80 else ""),
            "token_count"          : r_chunk.get("token_count", len(chunk.split())),
            "surprise_count"       : r_chunk.get("surprise_count", 0),
            "cached"               : False,
            "scoring_mode"         : scoring_mode,
            "probe_available"      : probe_available,
            "embedding_available"  : embedding_available,
            "provider"             : caps.get("provider", "unknown"),
            "weights_used"         : w_effective,
            "skipped_by_gate"      : True,
        }
        if use_cache and session is None:
            cache.set_cached_score(
                chunk, query, w_effective, skip_result,
                provider=_get_provider_for_cache(),
                model_version=_get_model_for_cache(),
            )
        if verbose:
            _print_chunk_report(skip_result, query, w_effective)
        return skip_result

    # Step 3 — relevance via cosine similarity
    cutoff = relevance_cutoff if relevance_cutoff is not None else DEFAULT_RELEVANCE_CUTOFF
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
            if cosine >= cutoff:
                rescaled        = (cosine - cutoff) / (1.0 - cutoff)
                relevance_score = max(0, min(100, round(rescaled * 100)))
            else:
                relevance_score = 0
        except Exception as e:
            # Suppress noisy warnings in batch contexts - only log if verbose
            if verbose:
                _logger.warning("embedding failed: %s — relevance set to 0", e)
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
    verdict, recommendation = _verdict(rag_score, thresholds)

    result = {
        "rag_score"            : rag_score,
        "novelty_score"        : novelty_score,
        "incremental_novelty"  : incremental_novelty,
        "conditional_novelty"  : conditional_novelty if conditional_novelty is not None else novelty_score,
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
        "weights_used"         : w_effective,
    }

    # Cache the result (only if not using session)
    # Cache key includes provider and model to prevent cross-provider collisions
    if use_cache and session is None:
        cache.set_cached_score(
            chunk, query, w_effective, result,
            provider=_get_provider_for_cache(),
            model_version=_get_model_for_cache(),
        )

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
    relevance_cutoff: float = None,
    thresholds: list = None,
) -> list:
    """
    Score multiple chunks and return them ranked by RAG usefulness (best first).

    New features:
      - incremental=True: maintain cross-chunk context via shared ModelSession
        (chunk B scored after chunk A's tokens fed through the model's KV cache)
      - diversity_threshold: dedup chunks that are >threshold cosine similar to already-selected ones
      - query ignorance gate: if model knows the answer well (<20 ignorance),
        all chunks are skipped automatically
      - weight renormalization: when probe unavailable, query_ignorance weight
        is redistributed to novelty and relevance proportionally

    Args:
        chunks              : list of text strings
        query               : the user query
        verbose             : print reports
        weights             : tunable scoring weights
        incremental         : enable cross-chunk incremental novelty tracking
                              (uses ModelSession to feed accepted chunk tokens between scores)
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
        if "error" not in r_query:
            query_knowledge = r_query["knowledge_score"]

    # ── Query ignorance gate: check if retrieval is even needed ──────────────
    if query_knowledge is not None and (100 - query_knowledge) < 20:
        # Model already knows the answer well — return empty early
        _logger.info("Query ignorance < 20%% — model already knows the answer. Skipping retrieval.")
        return []

    total = len(chunks)
    _logger.info("Scoring %d chunks (incremental=%s)...", total, incremental)

    # Embed all chunks once for diversity dedup
    chunk_embeddings: list = []
    if diversity_threshold and diversity_threshold < 1.0:
        try:
            chunk_embeddings = [embed(c) for c in chunks]
        except Exception:
            pass

    session = ModelSession() if incremental and probe is not None else None

    results = []
    for i, chunk in enumerate(chunks):
        _logger.debug("chunk %d/%d", i + 1, total)
        r = score_chunk(
            chunk, query=query, verbose=verbose,
            weights=weights, query_knowledge=query_knowledge,
            session=session, relevance_cutoff=relevance_cutoff,
            thresholds=thresholds,
        )
        r["original_index"] = i
        results.append(r)

    _apply_diversity_dedup(results, chunk_embeddings, diversity_threshold)

    results.sort(key=lambda x: x.get("rag_score", 0), reverse=True)
    for rank, r in enumerate(results):
        r["rank"] = rank + 1

    _logger.info("Done scoring %d chunks.", total)
    return results


# ── Batched scoring (faster) ─────────────────────────────────────────────────


def score_chunks_batch(
    chunks: list,
    query: str = None,
    verbose: bool = False,
    weights: dict = None,
    diversity_threshold: float = 0.85,
    relevance_cutoff: float = None,
    thresholds: list = None,
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

    print("  computing scores...           \r")

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
        cutoff = relevance_cutoff if relevance_cutoff is not None else DEFAULT_RELEVANCE_CUTOFF
        relevance_score = 0
        if query and q_vec is not None and c_vec is not None:
            cosine = _cosine_similarity(q_vec, c_vec)
            if cosine >= cutoff:
                rescaled = (cosine - cutoff) / (1.0 - cutoff)
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
        verdict, recommendation = _verdict(rag_score, thresholds)

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

    _apply_diversity_dedup(results, chunk_embeddings, diversity_threshold)

    results.sort(key=lambda x: x.get("rag_score", 0), reverse=True)
    for rank, r in enumerate(results):
        r["rank"] = rank + 1

    print("[pymrsf.rag] Batch done.        ")
    return results


def explain_chunk(chunk: str, query: str = None, weights: dict = None) -> None:
    """Print a detailed explanation of why a chunk scores the way it does.

    Delegates to score_chunk with verbose=True for the human-readable report.

    Args:
        chunk: Text chunk to explain
        query: Query to score against (optional)
        weights: Scoring weights override (optional)

    Example:
        >>> explain_chunk("Backpropagation uses the chain rule.", query="How does backprop work?")
    """
    score_chunk(chunk, query=query, verbose=True, weights=weights)


# ── Printer ───────────────────────────────────────────────────────────────────


def _print_chunk_report(result: dict, query: str = None, weights: dict = None) -> None:
    w = weights or DEFAULT_WEIGHTS
    bar_len = 30

    def bar(score):
        filled = round(score / 100 * bar_len)
        return "█" * filled + "░" * (bar_len - filled)

    print(f"\n{'═' * 65}")
    print("  PYMRSF RAG CHUNK SCORER")
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
    relevance_cutoff    : float = None,
    thresholds          : list = None,
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
        relevance_cutoff=relevance_cutoff,
        thresholds=thresholds,
    )
    passed  = [r for r in scored if r["rag_score"] >= min_rag_score]
    dropped = len(scored) - len(passed)

    if top_k:
        passed = passed[:top_k]

    if verbose:
        print(f"\n{'═' * 65}")
        print("  PYMRSF CHUNK FILTER")
        print(f"{'═' * 65}")
        print(f"  Query        : {query[:60]}")
        print(f"  Input chunks : {len(chunks)}")
        print(f"  Min score    : {min_rag_score}/100")
        label = f"on (>{diversity_threshold * 100:.0f}% similar = dedup)" if diversity_threshold < 1.0 else "off"
        print(f"  Diversity    : {label}")
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


# ── Adaptive retrieval budget ─────────────────────────────────────────────────

_DEFAULT_IGNORANCE_BUDGET = {
    "high":   (70, None),  # ignorance >70% → return up to all passing chunks
    "medium": (40, 5),     # ignorance 40-70% → max 5 chunks
    "low":    (20, 2),     # ignorance 20-40% → max 2 chunks
    "none":   (0,  0),     # ignorance <20% → model already knows; skip retrieval
}


def smart_filter(
    chunks: list,
    query: str,
    min_score: int = 40,
    ignorance_budget: dict = None,
    diversity_threshold: float = 0.85,
    weights: dict = None,
    relevance_cutoff: float = None,
    thresholds: list = None,
    verbose: bool = False,
) -> dict:
    """Adaptive drop-in replacement for filter_chunks().

    Uses the model's query_ignorance score to decide how many chunks are
    actually needed — returning fewer when the model already knows the answer,
    more when it is highly ignorant.

    Args:
        chunks            : List of text strings to score and filter
        query             : The user query
        min_score         : Minimum rag_score to keep a chunk (default 40)
        ignorance_budget  : Dict mapping budget level to (ignorance_threshold, max_chunks).
                            Defaults to _DEFAULT_IGNORANCE_BUDGET.
        diversity_threshold: Cosine dedup threshold
        weights           : Custom scoring weights
        relevance_cutoff  : Minimum cosine similarity for relevance
        thresholds        : Custom verdict thresholds
        verbose           : Print budget decision

    Returns:
        {
            "chunks"          : list[str],   # selected chunk strings
            "query_ignorance" : int,         # 0-100 ignorance score
            "budget_applied"  : str,         # "high"/"medium"/"low"/"none"
            "skipped_reason"  : str | None,  # set when budget_applied=="none"
            "scored"          : list[dict],  # full scoring results for inspection
        }

    Example:
        result = smart_filter(chunks, query="What is RLHF?")
        if not result["chunks"]:
            print(result["skipped_reason"])
        else:
            answer = llm.complete(query, context=result["chunks"])
    """
    budget = ignorance_budget or _DEFAULT_IGNORANCE_BUDGET

    # Score all chunks
    scored = score_chunks(
        chunks, query=query, weights=weights,
        diversity_threshold=diversity_threshold,
        relevance_cutoff=relevance_cutoff,
        thresholds=thresholds,
    )
    if not scored:
        return {"chunks": [], "query_ignorance": 0, "budget_applied": "none",
                "skipped_reason": "No chunks to score.", "scored": []}

    # Extract query_ignorance from first result (computed once for all chunks)
    query_ignorance = scored[0].get("query_ignorance", 0)

    # Determine budget level (sorted by threshold descending)
    sorted_levels = sorted(budget.items(), key=lambda kv: kv[1][0], reverse=True)
    budget_level = "none"
    max_chunks = 0
    for level, (threshold, cap) in sorted_levels:
        if query_ignorance >= threshold:
            budget_level = level
            max_chunks = cap  # None means no cap
            break

    if budget_level == "none" or max_chunks == 0:
        reason = (
            f"Model already knows the answer well (query_ignorance={query_ignorance}/100 < "
            f"{budget.get('none', (20, 0))[0]}%) — no retrieval needed."
        )
        if verbose:
            print(f"[smart_filter] {reason}")
        return {"chunks": [], "query_ignorance": query_ignorance,
                "budget_applied": "none", "skipped_reason": reason, "scored": scored}

    passing = [r for r in scored if r["rag_score"] >= min_score]
    if max_chunks is not None:
        passing = passing[:max_chunks]

    if verbose:
        print(f"[smart_filter] query_ignorance={query_ignorance}/100 → budget={budget_level} "
              f"(max_chunks={max_chunks}) → {len(passing)} chunk(s) selected")

    return {
        "chunks": [r["chunk"] for r in passing],
        "query_ignorance": query_ignorance,
        "budget_applied": budget_level,
        "skipped_reason": None,
        "scored": scored,
    }


# ── Async versions ─────────────────────────────────────────────────────────────


async def _score_chunk_incremental_async(
    chunk: str,
    query: str,
    weights: dict,
    query_knowledge: int,
    session: ModelSession,
    session_lock: asyncio.Lock,
    relevance_cutoff: float,
    thresholds: list,
) -> dict:
    """Score one chunk while holding the session lock so KV cache is fed in order."""
    async with session_lock:
        return await asyncio.to_thread(
            score_chunk, chunk,
            query=query, verbose=False, weights=weights,
            query_knowledge=query_knowledge, session=session,
            use_cache=True, relevance_cutoff=relevance_cutoff,
            thresholds=thresholds,
        )


async def score_chunk_async(
    chunk: str,
    query: str = None,
    verbose: bool = False,
    weights: dict = None,
    query_knowledge: int = None,
    use_cache: bool = True,
    session: ModelSession = None,
    session_lock: asyncio.Lock = None,
    relevance_cutoff: float = None,
    thresholds: list = None,
) -> dict:
    """
    Async version of score_chunk - runs scoring in executor to avoid blocking.

    Args: Same as score_chunk, plus optional session/session_lock for incremental use.
    Returns: Same as score_chunk

    Example:
        import asyncio
        result = await score_chunk_async(chunk, query="...")
    """
    if session is not None and session_lock is not None:
        return await _score_chunk_incremental_async(
            chunk, query, weights, query_knowledge,
            session, session_lock, relevance_cutoff, thresholds,
        )
    return await asyncio.to_thread(
        score_chunk, chunk,
        query=query, verbose=verbose, weights=weights,
        query_knowledge=query_knowledge, use_cache=use_cache,
        relevance_cutoff=relevance_cutoff, thresholds=thresholds,
    )


async def score_chunks_async(
    chunks: list[str],
    query: str = None,
    verbose: bool = False,
    weights: dict = None,
    diversity_threshold: float = 0.85,
    max_concurrent: int = 10,
    incremental: bool = False,
    relevance_cutoff: float = None,
    thresholds: list = None,
) -> list[dict]:
    """
    Async version that scores multiple chunks concurrently.

    Args:
        chunks: List of text strings
        query: The user query
        verbose: Print reports
        weights: Tunable scoring weights
        diversity_threshold: Cosine dedup threshold
        max_concurrent: Maximum concurrent tasks (ignored when incremental=True)
        incremental: Feed each chunk into the shared KV cache in order so that
                     later chunks are scored in the context of earlier ones.
                     Requires local provider. Falls back to non-incremental silently.
        relevance_cutoff: Minimum cosine similarity to count as relevant (default 0.30)
        thresholds: Custom verdict thresholds list

    Returns:
        List of result dicts ranked by rag_score (best first)

    Example:
        import asyncio
        results = await score_chunks_async(chunks, query="...", incremental=True)
    """
    if not chunks:
        return []

    # Pre-compute query knowledge once
    query_knowledge = None
    if query and probe is not None:
        r_query = await asyncio.to_thread(probe, query)
        if "error" not in r_query:
            query_knowledge = r_query["knowledge_score"]

    # ── Incremental path: sequential, shared session + lock ──────────────────
    caps = provider_capabilities()
    use_incremental = incremental and caps.get("supports_probe", False)

    if use_incremental:
        print(f"\n[pymrsf.rag] Async scoring {len(chunks)} chunks (incremental=True, sequential)...")
        session = ModelSession()
        session_lock = asyncio.Lock()
        results = []
        for i, chunk in enumerate(chunks):
            print(f"  chunk {i+1}/{len(chunks)}...", end="\r")
            result = await _score_chunk_incremental_async(
                chunk, query, weights, query_knowledge,
                session, session_lock, relevance_cutoff, thresholds,
            )
            result["original_index"] = i
            results.append(result)
        results_list = results
    else:
        # ── Concurrent path ──────────────────────────────────────────────────
        print(f"\n[pymrsf.rag] Async scoring {len(chunks)} chunks (max_concurrent={max_concurrent})...")
        semaphore = asyncio.Semaphore(max_concurrent)

        async def score_with_limit(i: int, chunk: str):
            async with semaphore:
                result = await score_chunk_async(
                    chunk=chunk, query=query, verbose=verbose,
                    weights=weights, query_knowledge=query_knowledge,
                    relevance_cutoff=relevance_cutoff, thresholds=thresholds,
                )
                result["original_index"] = i
                return result

        tasks = [score_with_limit(i, chunk) for i, chunk in enumerate(chunks)]
        results_list = list(await asyncio.gather(*tasks))

    # Diversity dedup — collect embeddings from cache (already populated during scoring)
    chunk_embeddings_async: list = []
    if diversity_threshold and diversity_threshold < 1.0:
        for chunk in chunks:
            cached_emb = cache.get_cached_embedding(chunk)
            if cached_emb is not None:
                chunk_embeddings_async.append(cached_emb)
            else:
                try:
                    emb = embed(chunk)
                    cache.set_cached_embedding(chunk, emb)
                    chunk_embeddings_async.append(emb)
                except Exception:
                    chunk_embeddings_async.append(None)

    _apply_diversity_dedup(results_list, chunk_embeddings_async, diversity_threshold)

    results_list.sort(key=lambda x: x.get("rag_score", 0), reverse=True)
    for rank, r in enumerate(results_list):
        r["rank"] = rank + 1

    print("[pymrsf.rag] Async scoring done.")
    return results_list


async def filter_chunks_async(
    chunks: list[str],
    query: str,
    min_rag_score: int = 50,
    top_k: int = None,
    verbose: bool = False,
    weights: dict = None,
    diversity_threshold: float = 0.85,
    max_concurrent: int = 10,
    relevance_cutoff: float = None,
    thresholds: list = None,
) -> list[str]:
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
        relevance_cutoff=relevance_cutoff,
        thresholds=thresholds,
    )

    passed = [r for r in scored if r["rag_score"] >= min_rag_score]
    dropped = len(scored) - len(passed)

    if top_k:
        passed = passed[:top_k]

    if verbose:
        print(f"\n{'═' * 65}")
        print("  PYMRSF CHUNK FILTER (ASYNC)")
        print(f"{'═' * 65}")
        print(f"  Query        : {query[:60]}")
        print(f"  Input chunks : {len(chunks)}")
        print(f"  Min score    : {min_rag_score}/100")
        label = f"on (>{diversity_threshold * 100:.0f}% similar = dedup)" if diversity_threshold < 1.0 else "off"
        print(f"  Diversity    : {label}")
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
