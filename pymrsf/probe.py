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
import numpy as np

from .core import (
    MODEL_VERSION,
    detokenize,
    get_backend,
    get_raw_lm,
    provider_capabilities,
    quantized_argmax,
    tokenize,
)

# ── Thresholds ────────────────────────────────────────────────────────────────
# These were calibrated from experiments:
#   famous text (pangrams, Wikipedia, Bible) → 70–95%
#   AI-generated text                        → 60–85%
#   common conversational text               → 40–65%
#   novel / personal / proprietary text      → 10–40%

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


def probe(text: str, verbose: bool = False) -> dict:
    """
    Probe how well Mistral knows a piece of text.

    Returns:
        {
            "compression"    : float,   # 0.0 – 1.0 (higher = model knows it better)
            "knowledge_score": int,     # 0 – 100 (human-friendly version)
            "label"          : str,     # memorized / familiar / common / uncommon / unknown
            "description"    : str,     # plain-English explanation
            "token_count"    : int,
            "surprise_count" : int,
            "surprises"      : list,    # list of (position, token_str)
            "heatmap"        : list,    # list of {"token": str, "surprised": bool}
            "model"          : str,
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

    lm_obj.reset()
    lm_obj.eval(token_ids)

    surprises = []
    heatmap   = []

    for i in range(n - 1):
        pred_id   = quantized_argmax(np.array(lm_obj.scores[i]))
        actual_id = token_ids[i + 1]
        token_str = detokenize([actual_id]).strip() or f"<{actual_id}>"
        surprised = pred_id != actual_id

        if surprised:
            surprises.append((i + 1, token_str))

        heatmap.append({
            "token"    : token_str,
            "surprised": surprised,
            "position" : i + 1,
        })

    compression    = 1 - len(surprises) / max(n - 1, 1)
    knowledge_score = round(compression * 100)
    label, description = _label(compression)

    if verbose:
        _print_report(text, compression, knowledge_score, label, description,
                      surprises, heatmap, n)

    return {
        "compression"    : round(compression, 4),
        "knowledge_score": knowledge_score,
        "label"          : label,
        "description"    : description,
        "token_count"    : n,
        "surprise_count" : len(surprises),
        "surprises"      : surprises,
        "heatmap"        : heatmap,
        "model"          : MODEL_VERSION,
    }


def probe_compare(texts: list[str]) -> list[dict]:
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
    print("  PYMRSF KNOWLEDGE PROBE")
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
        print("  Surprise tokens (what the model didn't expect):")
        for pos, tok in surprises[:10]:
            print(f"    pos {pos:>3} → '{tok}'")
        if len(surprises) > 10:
            print(f"    ... and {len(surprises) - 10} more")
    else:
        print("  No surprises — model predicted every token perfectly.")

    print("\n  Token heatmap  (✅ predicted | ⚡ surprise):")
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
