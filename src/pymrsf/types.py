"""Typed result shapes for pymrsf.

These are ``TypedDict``s (plain ``dict`` at runtime), so they add editor
autocomplete and static checking for ``py.typed`` consumers without changing any
behaviour. ``total=False`` because the exact key set varies by path (e.g. the
query-ignorance skip path, relevance-only mode, or probe errors).
"""
from typing import List, Tuple, TypedDict


class ProbeResult(TypedDict, total=False):
    """Return shape of :func:`pymrsf.probe`."""
    compression: float       # argmax-based knowledge fraction (0..1)
    confidence: float        # geometric-mean P(actual token) (0..1)
    perplexity: float        # exp(-avg logprob), >= 1
    avg_logprob: float
    metric: str              # "argmax" | "confidence" (drives knowledge_score)
    knowledge_score: int     # 0..100
    label: str               # memorized | familiar | common | uncommon | unknown
    description: str
    token_count: int
    surprise_count: int
    surprises: List[Tuple[int, str]]
    heatmap: List[dict]
    model: str
    text: str                # added by probe_compare()
    error: str               # present only on failure
    message: str


class ScoreResult(TypedDict, total=False):
    """Return shape of :func:`pymrsf.score_chunk` and the batch/async scorers."""
    rag_score: int           # 0..100
    novelty_score: int
    incremental_novelty: int
    relevance_score: int
    knowledge_score: int
    query_knowledge: int
    query_ignorance: int
    verdict: str             # excellent | good | moderate | weak | skip
    recommendation: str
    chunk: str
    chunk_preview: str
    token_count: int
    surprise_count: int
    cached: bool
    scoring_mode: str        # "full" | "relevance_only"
    probe_available: bool
    embedding_available: bool
    provider: str
    weights_used: dict
    skipped_by_gate: bool    # set on the query-ignorance skip path
    original_index: int      # set by the multi-chunk scorers
    rank: int                # set by the multi-chunk scorers
