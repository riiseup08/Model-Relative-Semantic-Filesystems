# pymrsf — Model-Relative Semantic Filtering

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/riiseup08/Model-Relative-Semantic-Filesystems/blob/main/LICENSE)
[![CI](https://github.com/riiseup08/Model-Relative-Semantic-Filesystems/actions/workflows/ci.yml/badge.svg)](https://github.com/riiseup08/Model-Relative-Semantic-Filesystems/actions/workflows/ci.yml)
[![Tests](https://img.shields.io/badge/tests-84%20passing-brightgreen)]()

**Score RAG chunks by information gain — not just relevance.**

Vector databases and semantic chunkers retrieve by relevance (cosine similarity). A chunk can be highly relevant yet contain only facts the model already memorized during training — wasted context window. pymrsf uses the model's own predictive surprise to detect which chunks contain genuinely *new* information.

- **Novelty**: Does the model already know this? (surprise-based)
- **Relevance**: Is this related to the query? (cosine similarity)
- **Query Ignorance**: Does the model even know the answer? (probe-based gate)
- **Diversity**: Does a better chunk already cover this? (dedup post-filter)

---

## Quick install

```bash
# Start fast with an API provider (no 4 GB model download)
pip install pymrsf[openai]
export OPENAI_API_KEY='sk-...'

# Or for full features (probing, smart_chunk, round-trip):
pip install pymrsf[local]
```

All providers require [Ollama](https://ollama.ai/) for embeddings:

```bash
ollama pull nomic-embed-text
```

---

## 30-second example — score and filter chunks

```python
from pymrsf import score_chunk, filter_chunks

chunks = [
    "Backpropagation computes gradients using the chain rule.",
    "Neural networks are inspired by the human brain.",
    "The sky is blue because of Rayleigh scattering.",
]

# Score a single chunk
result = score_chunk(chunks[0], query="How does backpropagation work?")
print(result["rag_score"])   # 0–100
print(result["verdict"])     # "excellent" / "good" / "moderate" / "weak" / "skip"

# Filter to only the useful chunks
useful = filter_chunks(chunks, query="How does backpropagation work?", min_rag_score=50)
# useful ≈ ["Backpropagation computes gradients..."]
```

With **async** for production pipelines:

```python
import asyncio
from pymrsf import filter_chunks_async

useful = asyncio.run(filter_chunks_async(chunks, query="...", min_rag_score=50))
```

---

## 60-second example — surprise-guided chunking

Instead of splitting at fixed sizes or sentence boundaries, `smart_chunk` uses the model's surprise signal to find natural knowledge transitions:

```python
from pymrsf import smart_chunk

long_article = """
Quantum computing leverages superposition and entanglement to perform
calculations that would be infeasible for classical computers. Unlike
classical bits, qubits can exist in multiple states simultaneously.
...
Machine learning models learn patterns from data through iterative
optimization of a loss function. Neural networks, in particular,
use backpropagation to adjust millions of parameters.
...
"""

# Chunks split at the boundary between "quantum computing" and "ML" —
# where the model's surprise signal drops after absorbing one topic
pieces = smart_chunk(long_article, min_chunk_len=200, max_chunk_len=800)
```

**Requires the local provider.** Falls back to sentence splitting for API providers.

---

## Provider matrix

This is the most important table in this README — it tells you which features work with which provider.

| Feature | local | openai | anthropic |
|---------|-------|--------|-----------|
| **RAG scoring** | Full (novelty + relevance + ignorance) | Relevance-only | Relevance-only |
| **Knowledge probing** | ✅ Full | ⚠️ Limited | ❌ |
| **smart_chunk** (surprise-guided) | ✅ Yes | Fallback to sentence | Fallback to sentence |
| **Delta compression / round-trip** | ✅ Yes | ❌ | ❌ |
| **Model session** (KV-cache) | ✅ Yes | ❌ | ❌ |
| **Async scoring** | ✅ | ✅ | ✅ |
| **Score caching** | ✅ | ✅ | ✅ |

**Key takeaway:** probing, smart_chunk, and the experimental round-trip storage all require the **local** provider (`pip install pymrsf[local]` + a GGUF model). If you only need relevance-based RAG scoring, OpenAI or Anthropic work fine.

---

## Production configuration

```python
import pymrsf

# Enable pymrsf log output (silent by default)
pymrsf.configure_logging("INFO")

# Tweak runtime settings without touching env vars
pymrsf.configure(
    provider="openai",
    embed_timeout=60,
    default_relevance_cutoff=0.4,
)
```

Environment variables for container/CI environments:

```bash
PYMRSF_PROVIDER=openai
OPENAI_API_KEY=sk-...
PYMRSF_ALLOW_PROVIDER_FALLBACK=true   # silently fall back on embed failures
PYMRSF_EMBED_TIMEOUT=30
```

- `PYMRSF_ALLOW_PROVIDER_FALLBACK` — when `true`, embed failures log a warning and continue instead of raising. Off by default (fail-fast).
- `pymrsf.configure_logging("WARNING")` — pymrsf ships with a `NullHandler` so `import pymrsf` is silent until you opt in.

See [ENV_CONFIG.md](https://github.com/riiseup08/Model-Relative-Semantic-Filesystems/blob/main/ENV_CONFIG.md) for all supported variables.

---

## Experimental: MRSF delta-compression storage

The round-trip storage backend stores only "surprise" tokens (40–60% compression) and reconstructs text via O(n) model inference. Import from `pymrsf.experimental` to signal the research-grade scope:

```python
from pymrsf.experimental import mrsf_write, mrsf_read, save_index

doc = mrsf_write("The Eiffel Tower was built in 1889.")
print(doc["compression"])   # 0.47 — 47% of tokens were predictable

save_index()
results = mrsf_read("famous French landmark", top_k=1)
```

[Full experimental docs →](https://github.com/riiseup08/Model-Relative-Semantic-Filesystems/tree/main/pymrsf/experimental)

---

## Score interpretation

| Score | Verdict | Suggested action |
|-------|---------|-----------------|
| 80–100 | excellent | Prioritise |
| 60–79 | good | Include |
| 40–59 | moderate | Include if space allows |
| 20–39 | weak | Skip if better chunks exist |
| 0–19 | skip | Model already knows this |

---

## Additional documentation

- [PROVIDER_SUPPORT.md](https://github.com/riiseup08/Model-Relative-Semantic-Filesystems/blob/main/PROVIDER_SUPPORT.md) — full capability matrix with programmatic checks
- [ENV_CONFIG.md](https://github.com/riiseup08/Model-Relative-Semantic-Filesystems/blob/main/ENV_CONFIG.md) — all environment variables
- [docs/CONCURRENCY.md](https://github.com/riiseup08/Model-Relative-Semantic-Filesystems/blob/main/docs/CONCURRENCY.md) — threading and process-safety model
- [CHANGELOG.md](https://github.com/riiseup08/Model-Relative-Semantic-Filesystems/blob/main/CHANGELOG.md) — version history

## Paper

The technical approach is described in the MRSF paper (link forthcoming). For now, see [CHANGELOG.md](https://github.com/riiseup08/Model-Relative-Semantic-Filesystems/blob/main/CHANGELOG.md) for the research lineage and [the experimental module](https://github.com/riiseup08/Model-Relative-Semantic-Filesystems/tree/main/pymrsf/experimental) for the delta-compression implementation.

## License

MIT
