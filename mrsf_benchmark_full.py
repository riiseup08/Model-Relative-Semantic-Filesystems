"""
MRSF Full Canterbury Corpus Benchmark -- Chunked Eval Fix
---------------------------------------------------------
Root cause of previous crashes: llama_decode returned 1
= KV cache overflow when feeding all tokens at once.

Fix: process files in sliding windows of CHUNK_SIZE tokens,
resetting the KV cache between chunks. This trades a small
amount of cross-chunk context for the ability to process
arbitrarily long files on any hardware.

Usage:
    python mrsf_benchmark_full.py 2>&1 | tee mrsf_benchmark.log
"""

import os, sys, time, csv, traceback

# Force UTF-8 output on Windows (avoids cp1252 UnicodeEncodeError)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
from datetime import datetime
from pathlib import Path
import numpy as np

# -- CONFIG ------------------------------------------------------------------
GGUF_PATH       = "./models/mistral-7b-v0.1.Q4_K_M.gguf"
CORPUS_DIR      = "./canterbury"
RESULTS_CSV     = "./mrsf_results_full.csv"
LOGIT_PRECISION = 6
N_CTX           = 2048        # safe conservative value -- fits in RAM
CHUNK_SIZE      = 1024        # tokens per chunk (must be < N_CTX)
CHUNK_OVERLAP   = 64          # overlap tokens for context continuity
MAX_TOKENS      = 150_000     # absolute cap (kennedy.xls has 1M tokens)
N_THREADS       = os.cpu_count()
# ----------------------------------------------------------------------------

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

# -- Load model ---------------------------------------------------------------
log("Loading model...")
from llama_cpp import Llama

lm = Llama(
    model_path   = GGUF_PATH,
    n_ctx        = N_CTX,
    n_gpu_layers = 0,
    logits_all   = True,
    verbose      = False,
    n_threads    = N_THREADS,
)
log(f"Model loaded. {N_THREADS} threads. n_ctx={N_CTX}, chunk={CHUNK_SIZE}")

# -- Helpers ------------------------------------------------------------------
def tokenize(text):
    return lm.tokenize(text.encode("utf-8", errors="replace"), add_bos=True)

def quantized_argmax(raw_logits):
    arr = np.array(raw_logits, dtype=np.float64)
    return int(np.argmax(np.round(arr, decimals=LOGIT_PRECISION)))

def read_file(path):
    for enc in ("utf-8", "latin-1"):
        try:
            return Path(path).read_text(encoding=enc, errors="replace")
        except Exception:
            continue
    return None

def load_done():
    done = set()
    if os.path.exists(RESULTS_CSV):
        with open(RESULTS_CSV, newline="") as f:
            for row in csv.DictReader(f):
                done.add(row["file"])
    return done

def append_result(row):
    write_header = not os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            w.writeheader()
        w.writerow(row)

def count_surprises_chunked(token_ids):
    """
    Process token_ids in chunks of CHUNK_SIZE.
    Overlaps by CHUNK_OVERLAP tokens between chunks for context continuity.
    Only scores predictions in the non-overlap region per chunk.
    Returns (delta, scored).
    """
    delta     = 0
    scored    = 0
    n         = len(token_ids)
    pos       = 0
    chunk_num = 0

    while pos < n - 1:
        chunk_end = min(pos + CHUNK_SIZE, n)
        chunk     = token_ids[pos:chunk_end]

        if len(chunk) < 2:
            break

        lm.reset()
        lm.eval(chunk)

        # Skip overlap region on all chunks except the first
        score_start = CHUNK_OVERLAP if pos > 0 else 0

        for i in range(score_start, len(chunk) - 1):
            pred   = quantized_argmax(np.array(lm.scores[i]))
            actual = chunk[i + 1]
            if pred != actual:
                delta += 1
            scored += 1

        chunk_num += 1
        if chunk_num % 10 == 0:
            pct = 100 * chunk_end / n
            log(f"    chunk {chunk_num} | {chunk_end:,}/{n:,} tokens ({pct:.0f}%) | delta={delta}")

        # Advance with overlap
        next_pos = chunk_end - CHUNK_OVERLAP
        if next_pos <= pos:
            next_pos = chunk_end  # safety: never stall
        pos = next_pos

    return delta, scored

# -- Main loop ----------------------------------------------------------------
files = sorted(p for p in Path(CORPUS_DIR).iterdir() if p.is_file())
done  = load_done()

log(f"Found {len(files)} files. Already done: {sorted(done) or 'none'}")

for fpath in files:
    fname = fpath.name

    if fname in done:
        log(f"SKIP {fname} (already in CSV)")
        continue

    log("-" * 55)
    log(f"START  {fname}  ({fpath.stat().st_size:,} bytes)")

    text = read_file(fpath)
    if not text or len(text.strip()) < 5:
        log("  SKIP -- empty or unreadable")
        continue

    char_count = len(text)

    try:
        all_tokens = tokenize(text)
    except Exception as e:
        log(f"  TOKENIZE ERROR: {e}")
        continue

    n_raw = len(all_tokens)
    log(f"  Chars: {char_count:,} | Tokens: {n_raw:,}")

    capped = False
    if n_raw > MAX_TOKENS:
        log(f"  Capping at {MAX_TOKENS:,} tokens")
        all_tokens = all_tokens[:MAX_TOKENS]
        char_count = int(char_count * MAX_TOKENS / n_raw)
        capped     = True

    n_tokens = len(all_tokens)
    t0 = time.time()

    try:
        delta, scored = count_surprises_chunked(all_tokens)
    except Exception as e:
        log(f"  CRASH: {e}")
        traceback.print_exc()
        continue

    t_total     = time.time() - t0
    compression = 1.0 - delta / max(scored, 1)
    bpc         = (delta * 32) / (char_count * 8) if char_count > 0 else 0.0

    log(f"  delta={delta} | scored={scored:,} | compression={compression:.1%} | BPC={bpc:.4f} | {t_total:.0f}s")

    row = {
        "file"        : fname,
        "bytes"       : fpath.stat().st_size,
        "chars"       : char_count,
        "tokens"      : n_tokens,
        "scored"      : scored,
        "delta"       : delta,
        "compression" : f"{compression:.4f}",
        "bpc"         : f"{bpc:.4f}",
        "time_s"      : f"{t_total:.1f}",
        "capped"      : "yes" if capped else "no",
        "timestamp"   : datetime.now().isoformat(),
    }
    append_result(row)
    log(f"  Saved -> {RESULTS_CSV}")

# -- Summary ------------------------------------------------------------------
log("=" * 55)
log("COMPLETE -- SUMMARY")
log("=" * 55)

all_rows = []
if os.path.exists(RESULTS_CSV):
    with open(RESULTS_CSV, newline="") as f:
        all_rows = list(csv.DictReader(f))

print(f"\n{'File':<20} {'Tokens':>8} {'Scored':>8} {'delta':>7} {'Compress':>10} {'BPC':>8} {'Time':>8}")
print("-" * 75)

bpcs, compressions = [], []
for r in sorted(all_rows, key=lambda x: x["file"]):
    b = float(r["bpc"])
    c = float(r["compression"])
    bpcs.append(b)
    compressions.append(c)
    cap = " *" if r.get("capped") == "yes" else ""
    print(f"{r['file']:<20} {int(r['tokens']):>8,} {int(r['scored']):>8,} "
          f"{int(r['delta']):>7,} {c:>9.1%} {b:>8.4f} {r['time_s']:>7}s{cap}")

if bpcs:
    print("-" * 75)
    print(f"{'AVERAGE':<20} {'':>8} {'':>8} {'':>7} "
          f"{sum(compressions)/len(compressions):>9.1%} "
          f"{sum(bpcs)/len(bpcs):>8.4f}")

print(f"\n* = token-capped at {MAX_TOKENS:,}")
print(f"Results: {RESULTS_CSV}")
