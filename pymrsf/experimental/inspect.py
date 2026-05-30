import numpy as np

from ..core import (
    ModelSession,
    compute_delta,
    detokenize,
    get_backend,
    get_raw_lm,
    provider_capabilities,
    quantized_argmax,
    tokenize,
)


def mrsf_inspect(text: str, return_data: bool = False):
    """Inspect delta compression for a text document.

    Args:
        text: Text to inspect
        return_data: If True, return structured dict instead of printing

    Returns:
        dict if return_data=True, containing:
          - surprises: list of surprise token strings
          - compression: float (0-1)
          - token_count: int
          - surprise_count: int
    """
    # Check if inspection is available
    if not provider_capabilities().get("supports_logits", False):
        error_msg = (
            "[ERROR] mrsf_inspect requires the local provider with raw model access.\n"
            "  Install with: pip install pymrsf[local]\n"
            "  And set: PYMRSF_PROVIDER=local"
        )
        if return_data:
            return {"error": error_msg}
        print(error_msg)
        return

    token_ids = tokenize(text)
    n         = len(token_ids)
    {pos: tid for pos, tid in compute_delta(token_ids)}

    # Need raw scores for display — load model and eval
    backend = get_backend()
    lm_obj = backend.get("lm") or get_raw_lm()
    if lm_obj:
        lm_obj.reset()
        lm_obj.eval(token_ids)
    else:
        error_msg = "[ERROR] mrsf_inspect requires a local provider."
        if return_data:
            return {"error": error_msg}
        print(error_msg)
        return

    print(f"\n{'─'*65}")
    print(f"Document : {text[:80]}")
    print(f"{'─'*65}")
    print(f"{'POS':<5} {'ACTUAL':<25} {'PREDICTED':<25} {'STATUS'}")
    print(f"{'─'*65}")

    surprises = []
    for i in range(n - 1):
        pred_id    = quantized_argmax(np.array(lm_obj.scores[i]))
        actual_id  = token_ids[i + 1]
        actual_str = detokenize([actual_id]).strip() or f"<id:{actual_id}>"
        pred_str   = detokenize([pred_id]).strip()   or f"<id:{pred_id}>"
        tag        = "⚡ SURPRISE" if pred_id != actual_id else "✅ predicted"
        if pred_id != actual_id:
            surprises.append(actual_str)
        print(f"{i+1:<5} {actual_str:<25} {pred_str:<25} {tag}")

    print(f"{'─'*65}")
    print(f"Surprise tokens in Δ : {surprises}")
    print(f"Compression          : {1 - len(surprises) / max(n-1, 1):.1%}\n")

    # Return structured data if requested
    if return_data:
        return {
            "surprises": surprises,
            "compression": 1 - len(surprises) / max(n-1, 1),
            "token_count": n,
            "surprise_count": len(surprises),
        }


def mrsf_rebuild_explained(text: str, return_data: bool = False):
    """Explain step-by-step how delta reconstruction works.

    Args:
        text: Text to rebuild
        return_data: If True, return structured dict instead of printing

    Returns:
        dict if return_data=True, containing:
          - match: bool (whether rebuild matches original)
          - original: str
          - rebuilt: str
          - steps: list of (pos, source, token) tuples
    """
    # Check if rebuild is available
    if not provider_capabilities().get("supports_delta", False):
        error_msg = (
            "[ERROR] mrsf_rebuild_explained requires the local provider.\n"
            "  Install with: pip install pymrsf[local]\n"
            "  And set: PYMRSF_PROVIDER=local"
        )
        if return_data:
            return {"error": error_msg}
        print(error_msg)
        return

    token_ids = tokenize(text)
    n         = len(token_ids)

    delta_dict = {pos: tid for pos, tid in compute_delta(token_ids)}

    print(f"\n{'═'*65}")
    print(f"REBUILDING: {text[:70]}")
    print(f"{'═'*65}")
    print("\n STEP 1 — What Δ stores:")
    print(f"  {[(pos, detokenize([tid]).strip()) for pos, tid in delta_dict.items()]}")
    print("\n STEP 2 — Token by token reconstruction:\n")
    print(f"  {'POS':<5} {'SOURCE':<12} {'RUNNING TEXT'}")
    print(f"  {'─'*65}")

    bos     = tokenize("")[0]
    out_ids = [bos]
    session = ModelSession()
    session.feed(bos)

    steps = []  # For structured data return

    for i in range(1, n):
        if i in delta_dict:
            out_ids.append(delta_dict[i])
            source = "⚡ FROM Δ"
        else:
            out_ids.append(session.predict_next())
            source = "🤖 MODEL"
        session.feed(out_ids[-1])

        # Exclude BOS token when showing running text
        running = detokenize(out_ids[1:]).strip()
        token_str = detokenize([out_ids[-1]]).strip()
        steps.append((i, source, token_str))
        if not return_data:
            print(f"  {i:<5} {source:<12} {running[:60]}")

    # Exclude BOS token for final comparison
    rebuilt = detokenize(out_ids[1:]).strip()
    match = rebuilt == text.strip()

    if return_data:
        return {
            "match": match,
            "original": text,
            "rebuilt": rebuilt,
            "steps": steps,
        }

    print(f"\n{'═'*65}")
    print(f" ORIGINAL : {text}")
    print(f" REBUILT  : {rebuilt}")
    print(f" MATCH    : {'✅ Perfect reconstruction' if match else '⚠️  Minor tokenization diff'}")
    print(f"{'═'*65}\n")
