"""
pymrsf.probe — Knowledge probing: how well does the model know a text?

Given a text, measures compression rate = 1 - (surprise_tokens / total_tokens).
Higher compression = model knows it better.

**Provider Support**:
  - ✅ Local: Full knowledge probing with token-level surprises
  - ❌ OpenAI: Probing not supported (use relevance-based RAG instead)
  - ❌ Anthropic: Probing not supported (use relevance-based RAG instead)

**When probing is unavailable**:
  Non-local providers can still use `score_chunk()` and `filter_chunks()` from 
  the rag module for relevance-based RAG scoring without novelty detection.
  
  Example:
    >>> from pymrsf import score_chunk
    >>> result = score_chunk(chunk, query)  # Works with all providers
    >>> # Local: full novelty-aware scoring
    >>> # OpenAI/Anthropic: relevance-only scoring

**Usage**:
    >>> from pymrsf import probe, provider_capabilities
    >>> 
    >>> # Check if probing is available
    >>> if provider_capabilities()["supports_probe"]:
    ...     result = probe("The quick brown fox jumps over the lazy dog.")
    ...     print(result["knowledge_score"])  # 0-100
    ...     print(result["label"])  # memorized/familiar/common/uncommon/unknown
"""
import math
import os
from typing import List
import numpy as np
from .core import tokenize, detokenize, quantized_argmax, get_backend, get_raw_lm, MODEL_VERSION, provider_capabilities, model_lock
from .types import ProbeResult


# ── Thresholds ────────────────────────────────────────────────────────────────
# IMPORTANT: these cutoffs are MODEL-SPECIFIC. They were calibrated against the
# argmax "compression" metric on Mistral-7B (Q4_K_M):
#   famous text (pangrams, Wikipedia, Bible) → 70–95%
#   AI-generated text                        → 60–85%
#   common conversational text               → 40–65%
#   novel / personal / proprietary text      → 10–40%
#
# Pointing pymrsf at a different GGUF will shift these ranges, so the *labels*
# (memorized/familiar/…) may be miscalibrated even though the raw scores remain
# monotonic. If you change models:
#   - rely on relative ranking (novelty = 100 - knowledge_score) rather than the
#     absolute label, and/or use the model-agnostic confidence/perplexity signal
#     (PYMRSF_PROBE_METRIC=confidence), or
#   - override THRESHOLDS after recalibrating on your own reference texts.

THRESHOLDS = [
    (0.85, "memorized",    "Model has almost certainly seen this text verbatim."),
    (0.65, "familiar",     "Model knows this topic/style well — likely in training data."),
    (0.45, "common",       "Recognizable patterns but not memorized."),
    (0.25, "uncommon",     "Novel phrasing or topic — model finds this surprising."),
    (0.00, "unknown",      "Highly original or proprietary — model has little knowledge of this."),
]


def _label(compression: float) -> tuple[str, str]:
    for threshold, label, description in THRESHOLDS:
        if compression >= threshold:
            return label, description
    return "unknown", THRESHOLDS[-1][2]


def probe(text: str, verbose: bool = False) -> ProbeResult:
    """
    Probe how well the model knows a piece of text.

    Two knowledge signals are computed from the same forward pass:

      * argmax "compression" (default) — fraction of tokens the model would have
        predicted exactly via greedy argmax. Coarse/binary: a token that was the
        model's 2nd-likeliest at 49%% counts as a full surprise. This is what the
        THRESHOLDS table is calibrated against.
      * probability "confidence" — exp(mean log p(actual token)), i.e. the
        geometric-mean probability the model assigned to the real next token.
        Smoother and far less sensitive to near-ties; `perplexity` is its inverse.

    Set PYMRSF_PROBE_METRIC=confidence to drive knowledge_score/label/novelty
    from the smoother signal instead of argmax. Default ("argmax") preserves
    historical scores and calibration.

    Returns:
        {
            "compression"     : float,   # 0.0 – 1.0 argmax-based (higher = better known)
            "confidence"      : float,   # 0.0 – 1.0 geometric-mean token probability
            "perplexity"      : float,   # >= 1.0   (lower = better known)
            "avg_logprob"     : float,   # mean log p(actual token)
            "metric"          : str,     # which metric drives knowledge_score
            "knowledge_score" : int,     # 0 – 100 (from the selected metric)
            "label"           : str,     # memorized / familiar / common / uncommon / unknown
            "description"     : str,     # plain-English explanation
            "token_count"     : int,
            "surprise_count"  : int,
            "surprises"       : list,    # list of (position, token_str)  [argmax]
            "heatmap"         : list,    # list of {"token", "surprised", "position", "prob"}
            "model"           : str,
        }
    """
    # Check if probing is available
    if not provider_capabilities().get("supports_probe", False):
        return {
            "error": "Probing requires local provider with full model access",
            "message": (
                "\n[pymrsf] Knowledge probing requires the local provider.\n"
                "  Install with: pip install pymrsf[local]\n"
                "  And set: PYMRSF_PROVIDER=local\n"
                "  API providers (OpenAI, Anthropic) don't support this feature.\n"
            )
        }
    
    token_ids = tokenize(text)
    n         = len(token_ids)

    if n < 2:
        return {
            "error": "Text too short to probe",
            "message": "Text must contain at least 2 tokens for probing. Received text with 0-1 tokens."
        }

    # Get raw LM object for direct score access
    backend = get_backend()
    lm_obj  = backend.get("lm") or get_raw_lm()
    
    surprises = []
    heatmap   = []
    total_logprob = 0.0

    # The local model has a single mutable KV cache and is not reentrant; hold the
    # shared lock so concurrent callers can't corrupt each other's eval state.
    with model_lock():
        lm_obj.reset()
        lm_obj.eval(token_ids)

        for i in range(n - 1):
            logits    = np.asarray(lm_obj.scores[i], dtype=np.float64)
            pred_id   = quantized_argmax(logits)
            actual_id = token_ids[i + 1]

            # log-softmax probability of the *actual* next token (numerically
            # stable). This is the smooth signal that argmax throws away.
            m    = float(logits.max())
            logZ = m + math.log(float(np.exp(logits - m).sum()))
            actual_logprob = float(logits[actual_id]) - logZ
            total_logprob += actual_logprob

            token_str = detokenize([actual_id]).strip() or f"<{actual_id}>"
            surprised = pred_id != actual_id

            if surprised:
                surprises.append((i + 1, token_str))

            heatmap.append({
                "token"    : token_str,
                "surprised": surprised,
                "position" : i + 1,
                "prob"     : round(math.exp(actual_logprob), 4),
            })

    n_pred = max(n - 1, 1)
    compression = 1 - len(surprises) / n_pred          # argmax-based (calibrated)
    avg_logprob = total_logprob / n_pred
    confidence  = math.exp(avg_logprob)                # geometric-mean token prob, 0..1
    perplexity  = math.exp(-avg_logprob)

    metric = os.getenv("PYMRSF_PROBE_METRIC", "argmax").lower()
    knowledge_value = confidence if metric == "confidence" else compression
    knowledge_score = round(knowledge_value * 100)
    label, description = _label(knowledge_value)

    if verbose:
        _print_report(text, knowledge_value, knowledge_score, label, description,
                      surprises, heatmap, n)

    return {
        "compression"    : round(compression, 4),
        "confidence"     : round(confidence, 4),
        "perplexity"     : round(perplexity, 4),
        "avg_logprob"    : round(avg_logprob, 4),
        "metric"         : metric if metric == "confidence" else "argmax",
        "knowledge_score": knowledge_score,
        "label"          : label,
        "description"    : description,
        "token_count"    : n,
        "surprise_count" : len(surprises),
        "surprises"      : surprises,
        "heatmap"        : heatmap,
        "model"          : MODEL_VERSION,
    }


def probe_compare(texts: list[str]) -> List[ProbeResult]:
    """
    Probe multiple texts and return them ranked by knowledge score (highest first).
    
    Texts that fail to probe (e.g., too short or provider doesn't support probing)
    are included in results but sorted to the end with knowledge_score of -1.
    
    Args:
        texts: List of text strings to probe
        
    Returns:
        List of probe results, sorted by knowledge_score (descending)
        
    Example:
        >>> results = probe_compare([
        ...     "The quick brown fox",
        ...     "Neural networks learn by backpropagation",
        ...     "My secret proprietary algorithm XYZ-9000"
        ... ])
        >>> for r in results:
        ...     print(f"{r['text'][:30]}: {r['knowledge_score']}")
    """
    results = []
    for text in texts:
        r = probe(text)
        r["text"] = text
        
        # Handle error cases: set knowledge_score to -1 so they sort to the end
        if "error" in r and "knowledge_score" not in r:
            r["knowledge_score"] = -1
            
        results.append(r)

    # Sort by knowledge_score, errors (score=-1) go to the end
    results.sort(key=lambda x: x.get("knowledge_score", -1), reverse=True)
    return results


def _print_report(text, compression, score, label, description, surprises, heatmap, n):
    bar_len  = 30
    filled   = round(compression * bar_len)
    bar      = "█" * filled + "░" * (bar_len - filled)

    print(f"\n{'═' * 65}")
    print(f"  PYMRSF KNOWLEDGE PROBE")
    print(f"{'═' * 65}")
    print(f"  Text    : {text[:70]}{'...' if len(text) > 70 else ''}")
    print(f"  Model   : {MODEL_VERSION}")
    print(f"{'─' * 65}")
    print(f"  Score   : {score}/100  [{bar}]")
    print(f"  Label   : {label.upper()}")
    print(f"  Meaning : {description}")
    print(f"{'─' * 65}")
    print(f"  Tokens  : {n}  |  Surprises: {len(surprises)}  |  Compression: {compression:.1%}")
    print(f"{'─' * 65}")

    if surprises:
        print(f"  Surprise tokens (what the model didn't expect):")
        for pos, tok in surprises[:10]:
            print(f"    pos {pos:>3} → '{tok}'")
        if len(surprises) > 10:
            print(f"    ... and {len(surprises) - 10} more")
    else:
        print(f"  No surprises — model predicted every token perfectly.")

    print(f"\n  Token heatmap  (✅ predicted | ⚡ surprise):")
    line = ""
    for item in heatmap:
        mark = "⚡" if item["surprised"] else "✅"
        line += f"{mark}'{item['token']}' "
        if len(line) > 55:
            print(f"    {line}")
            line = ""
    if line:
        print(f"    {line}")

    print(f"{'═' * 65}\n")
