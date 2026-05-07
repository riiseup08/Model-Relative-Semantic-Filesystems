import numpy as np
from .core import lm, tokenize, detokenize, quantized_argmax, MODEL_VERSION


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
            "compression"   : float,   # 0.0 – 1.0 (higher = model knows it better)
            "knowledge_score": int,    # 0 – 100 (human-friendly version of compression)
            "label"         : str,     # memorized / familiar / common / uncommon / unknown
            "description"   : str,     # plain-English explanation
            "token_count"   : int,
            "surprise_count": int,
            "surprises"     : list,    # list of (position, token_str) that surprised the model
            "heatmap"       : list,    # list of {"token": str, "surprised": bool} for visualization
            "model"         : str,
        }
    """
    token_ids = tokenize(text)
    n         = len(token_ids)

    if n < 2:
        return {"error": "Text too short to probe (need at least 2 tokens)."}

    lm.reset()
    lm.eval(token_ids)

    surprises = []
    heatmap   = []

    for i in range(n - 1):
        pred_id   = quantized_argmax(np.array(lm.scores[i]))
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
    Each result includes the original text for reference.
    """
    results = []
    for text in texts:
        r = probe(text)
        r["text"] = text
        results.append(r)

    results.sort(key=lambda x: x["knowledge_score"], reverse=True)
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
