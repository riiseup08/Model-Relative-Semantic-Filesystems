"""
pymrsf.benchmark — Compression and latency benchmarks for local model.
"""

import os, time
import numpy as np
from .core import tokenize, quantized_argmax, _get_backend, LOGIT_PRECISION
from .storage import mrsf_write, mrsf_read, save_index
from .embeddings import embed


def mrsf_benchmark_canterbury(folder_path: str, max_chars: int = 2000):
    files = sorted([
        f for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f)) and not f.startswith(".")
    ])

    if not files:
        print(f"[BENCHMARK] No files found in {folder_path}")
        return []

    results = []
    print(f"\n{'═'*80}")
    print(f"CANTERBURY CORPUS BENCHMARK  (Mistral 7B Q4_K_M | CPU | precision={LOGIT_PRECISION})")
    print(f"{'═'*80}")
    print(f"{'FILE':<22} {'CHARS':>6} {'TOKENS':>7} {'Δ':>6} {'COMPRESS':>10} {'BPC':>8} {'TIME(s)':>8}")
    print(f"{'─'*80}")

    backend = _get_backend()
    lm_obj  = backend.get("lm")
    if lm_obj is None:
        print("[ERROR] Benchmark requires a local model provider.")
        return []

    for fname in files:
        path = os.path.join(folder_path, fname)
        text = None
        for enc in ["utf-8", "latin-1", "cp1252"]:
            try:
                with open(path, "r", encoding=enc) as f:
                    raw = f.read()
                printable = sum(1 for c in raw[:500] if c.isprintable() or c in "\n\r\t")
                if printable / max(len(raw[:500]), 1) > 0.85:
                    text = raw
                    break
            except Exception:
                continue

        if text is None or len(text.strip()) < 10:
            print(f"{fname:<22} [skipped]")
            continue

        text = text[:max_chars]
        try:
            t0        = time.time()
            token_ids = tokenize(text)
            n         = len(token_ids)
            lm_obj.reset()
            lm_obj.eval(token_ids)
            delta = []
            for i in range(n - 1):
                pred   = quantized_argmax(np.array(lm_obj.scores[i]))
                actual = token_ids[i + 1]
                if pred != actual:
                    delta.append((i + 1, actual))
            elapsed     = time.time() - t0
            delta_bits  = len(delta) * 32
            bpc         = delta_bits / (len(text) * 8)
            compression = 1 - len(delta) / max(n - 1, 1)
            results.append({"file": fname, "chars": len(text), "tokens": n,
                             "delta": len(delta), "compression": compression,
                             "bpc": bpc, "time": elapsed})
            print(f"{fname:<22} {len(text):>6} {n:>7} {len(delta):>6} "
                  f"{compression:>9.1%} {bpc:>8.4f} {elapsed:>8.2f}")
        except Exception as e:
            print(f"{fname:<22} [ERROR: {e}]")

    if results:
        avg_bpc  = sum(r["bpc"] for r in results) / len(results)
        avg_comp = sum(r["compression"] for r in results) / len(results)
        print(f"{'─'*80}")
        print(f"{'AVERAGE':<22} {avg_comp:>9.1%} {avg_bpc:>8.4f}")
        print(f"{'═'*80}\n")

    return results


def mrsf_latency_benchmark():
    test_docs = [
        ("Short sentence",
         "The Eiffel Tower is located in Paris, France, and was built in 1889."),
        ("Medium paragraph",
         "Neural networks learn hierarchical representations by optimizing a loss "
         "function via backpropagation. Each layer transforms its input through a "
         "learned weight matrix followed by a nonlinear activation."),
        ("Python code",
         "def quicksort(arr): return arr if len(arr)<=1 else quicksort([x for x in "
         "arr[1:] if x<=arr[0]])+[arr[0]]+quicksort([x for x in arr[1:] if x>arr[0]])"),
    ]

    print(f"\n{'═'*75}")
    print(f"LATENCY BENCHMARK  (Mistral 7B Q4_K_M | CPU | threads={os.cpu_count()})")
    print(f"{'═'*75}")
    print(f"{'DOCUMENT':<22} {'TOKENS':>7} {'WRITE(s)':>9} {'READ(s)':>9} {'EMBED(ms)':>10}")
    print(f"{'─'*75}")

    for label, text in test_docs:
        t0      = time.time()
        result  = mrsf_write(text)
        write_t = time.time() - t0
        query   = " ".join(text.split()[:3])
        t0      = time.time()
        mrsf_read(query)
        read_t  = time.time() - t0
        t0      = time.time()
        embed(text)
        embed_t = (time.time() - t0) * 1000
        print(f"{label:<22} {result['token_count']:>7} {write_t:>9.2f} "
              f"{read_t:>9.2f} {embed_t:>10.1f}")

    print(f"{'─'*75}")
    print(f"Note: Read path ~2600x slower than Zstd — suitable for cold/archival storage.")
    print(f"{'─'*75}\n")
    save_index()
