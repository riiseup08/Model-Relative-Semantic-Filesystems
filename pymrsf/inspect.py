import numpy as np
from .core import lm, tokenize, detokenize, quantized_argmax, ModelSession


def mrsf_inspect(text: str):
    token_ids = tokenize(text)
    n         = len(token_ids)
    lm.reset()
    lm.eval(token_ids)

    print(f"\n{'─'*65}")
    print(f"Document : {text[:80]}")
    print(f"{'─'*65}")
    print(f"{'POS':<5} {'ACTUAL':<25} {'PREDICTED':<25} {'STATUS'}")
    print(f"{'─'*65}")

    surprises = []
    for i in range(n - 1):
        pred_id    = quantized_argmax(np.array(lm.scores[i]))
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


def mrsf_rebuild_explained(text: str):
    token_ids = tokenize(text)
    n         = len(token_ids)
    lm.reset()
    lm.eval(token_ids)

    delta = {}
    for i in range(n - 1):
        pred_id   = quantized_argmax(np.array(lm.scores[i]))
        actual_id = token_ids[i + 1]
        if pred_id != actual_id:
            delta[i + 1] = actual_id

    print(f"\n{'═'*65}")
    print(f"REBUILDING: {text[:70]}")
    print(f"{'═'*65}")
    print(f"\n STEP 1 — What Δ stores:")
    print(f"  {[(pos, detokenize([tid]).strip()) for pos, tid in delta.items()]}")
    print(f"\n STEP 2 — Token by token reconstruction:\n")
    print(f"  {'POS':<5} {'SOURCE':<12} {'RUNNING TEXT'}")
    print(f"  {'─'*65}")

    bos     = tokenize("")[0]          # BOS token ID
    out_ids = [bos]
    session = ModelSession()
    session.feed(bos)

    for i in range(1, n):
        if i in delta:
            out_ids.append(delta[i])
            source = "⚡ FROM Δ"
        else:
            out_ids.append(session.predict_next())
            source = "🤖 MODEL"
        session.feed(out_ids[-1])

        # Exclude BOS token when showing running text
        running = detokenize(out_ids[1:]).strip()
        print(f"  {i:<5} {source:<12} {running[:60]}")

    # Exclude BOS token for final comparison
    rebuilt = detokenize(out_ids[1:]).strip()
    print(f"\n{'═'*65}")
    print(f" ORIGINAL : {text}")
    print(f" REBUILT  : {rebuilt}")
    match = rebuilt == text.strip()
    print(f" MATCH    : {'✅ Perfect reconstruction' if match else '⚠️  Minor tokenization diff'}")
    print(f"{'═'*65}\n")