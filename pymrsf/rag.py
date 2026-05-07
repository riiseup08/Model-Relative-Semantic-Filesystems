"""
pymrsf.rag — RAG chunk quality scorer

Core idea:
  A chunk is useful to RAG if it contains information the model doesn't
  already know AND that information is relevant to the query.

  - novelty_score   : how much NEW info (inverse of LLM knowledge)
  - relevance_score : cosine similarity between query and chunk embeddings
  - rag_score       : weighted combination — higher = more useful for RAG

Usage:
    from pymrsf.rag import score_chunk, score_chunks, explain_chunk

    result = score_chunk("Neural networks learn by...", "how does backprop work?")
    print(result["rag_score"])   # 0-100
    print(result["verdict"])     # excellent / good / moderate / weak / skip
"""

import numpy as np
from .probe import probe
from .embeddings import embed


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

def score_chunk(chunk: str, query: str = None, verbose: bool = False) -> dict:
    """
    Score a single RAG chunk for usefulness.

    Args:
        chunk   : the text chunk to evaluate
        query   : the user query (optional but recommended)
        verbose : print a human-readable report

    Returns:
        {
            "rag_score"       : int,   # 0-100, higher = more useful for RAG
            "novelty_score"   : int,   # how much NEW info (inverse of knowledge)
            "relevance_score" : int,   # cosine similarity to query (0 if no query)
            "knowledge_score" : int,   # how much model already knows this
            "verdict"         : str,   # excellent / good / moderate / weak / skip
            "recommendation"  : str,   # plain English
            "chunk_preview"   : str,
            "token_count"     : int,
            "surprise_count"  : int,
        }
    """
    # Step 1 — probe chunk: how much does the model already know this?
    r_chunk = probe(chunk)
    if "error" in r_chunk:
        return {"error": r_chunk["error"]}

    knowledge_score = r_chunk["knowledge_score"]
    novelty_score   = 100 - knowledge_score

    # Step 2 — relevance via cosine similarity between query and chunk embeddings
    # RELEVANCE_CUTOFF: cosine scores below this are treated as 0 (off-topic)
    # nomic-embed-text typically scores 0.50+ for genuinely related text
    RELEVANCE_CUTOFF = 0.30
    relevance_score = 0
    if query:
        try:
            q_vec  = embed(query)
            c_vec  = embed(chunk)
            cosine = _cosine_similarity(q_vec, c_vec)
            if cosine >= RELEVANCE_CUTOFF:
                # rescale from [cutoff, 1.0] → [0, 100] for cleaner spread
                rescaled        = (cosine - RELEVANCE_CUTOFF) / (1.0 - RELEVANCE_CUTOFF)
                relevance_score = max(0, min(100, round(rescaled * 100)))
            else:
                relevance_score = 0  # below cutoff = off-topic
        except Exception as e:
            print(f"  [warn] embedding failed: {e} — relevance set to 0")
            relevance_score = 0

    # Step 3 — combine into final RAG score
    # Novelty (60%) + Relevance (40%)
    # Both matter: a relevant chunk the model already knows is useless
    # A novel chunk that is off-topic is also useless
    if query:
        rag_score = round(novelty_score * 0.6 + relevance_score * 0.4)
    else:
        rag_score = novelty_score

    rag_score = max(0, min(100, rag_score))
    verdict, recommendation = _verdict(rag_score)

    result = {
        "rag_score"       : rag_score,
        "novelty_score"   : novelty_score,
        "relevance_score" : relevance_score,
        "knowledge_score" : knowledge_score,
        "verdict"         : verdict,
        "recommendation"  : recommendation,
        "chunk"           : chunk,
        "chunk_preview"   : chunk[:80] + ("..." if len(chunk) > 80 else ""),
        "token_count"     : r_chunk["token_count"],
        "surprise_count"  : r_chunk["surprise_count"],
    }

    if verbose:
        _print_chunk_report(result, query)

    return result


def score_chunks(chunks: list, query: str = None, verbose: bool = False) -> list:
    """
    Score multiple chunks and return them ranked by RAG usefulness (best first).
    """
    results = []
    total   = len(chunks)

    print(f"\n[pymrsf.rag] Scoring {total} chunks...")

    for i, chunk in enumerate(chunks):
        print(f"  chunk {i+1}/{total}...", end="\r")
        r = score_chunk(chunk, query=query, verbose=verbose)
        r["original_index"] = i
        results.append(r)

    results.sort(key=lambda x: x.get("rag_score", 0), reverse=True)

    for rank, r in enumerate(results):
        r["rank"] = rank + 1

    print(f"[pymrsf.rag] Done.{' ' * 20}")
    return results


def explain_chunk(chunk: str, query: str = None) -> None:
    """
    Print a detailed explanation of why a chunk scores the way it does.
    """
    score_chunk(chunk, query=query, verbose=True)


# ── Printer ───────────────────────────────────────────────────────────────────

def _print_chunk_report(result: dict, query: str = None) -> None:
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
    print(f"  Known by LLM {result['knowledge_score']:>3}/100  [{bar(result['knowledge_score'])}]")
    print(f"{'─' * 65}")
    print(f"  Verdict : {result['verdict'].upper()}")
    print(f"  Action  : {result['recommendation']}")
    print(f"  Tokens  : {result['token_count']}  |  Surprises: {result['surprise_count']}")
    print(f"{'═' * 65}\n")


# ── Pipeline filter ───────────────────────────────────────────────────────────

def filter_chunks(
    chunks      : list,
    query       : str,
    min_rag_score : int = 50,
    top_k       : int  = None,
    verbose     : bool = False,
) -> list:
    """
    Drop-in filter for RAG pipelines.
    Returns only the chunks worth sending to the LLM.

    Args:
        chunks        : list of text strings (your retrieved chunks)
        query         : the user query
        min_rag_score : minimum score to keep a chunk (default 50)
        top_k         : if set, return only the top K chunks after filtering
        verbose       : print a summary report

    Returns:
        list of chunk strings that passed the filter, ranked best first

    Example:
        from pymrsf.rag import filter_chunks

        chunks = retriever.get(query, top_k=20)
        good   = filter_chunks(chunks, query, min_rag_score=50, top_k=5)
        answer = llm.complete(query, context=good)
    """
    scored  = score_chunks(chunks, query=query)
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
