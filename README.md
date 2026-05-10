# pymrsf — Model-Relative Semantic Filtering

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-84%20passing-brightgreen)]()

**Stop wasting context window on information your LLM already knows.**

`pymrsf` scores RAG chunks by measuring **information gain** — not just relevance. It uses the
model's own predictive surprise to detect which chunks contain genuinely new information.

---

## Quick Start

```bash
pip install pymrsf[openai]
export OPENAI_API_KEY='sk-...'
```

```python
import pymrsf

chunks = [
    "Backpropagation computes gradients using the chain rule.",
    "Neural networks are inspired by the human brain.",
    "The sky is blue because of Rayleigh scattering.",
]

# Filter chunks to the ones that matter for this query
useful = pymrsf.filter_chunks(chunks, query="How does backpropagation work?", min_rag_score=50)

# Surprise-guided chunking — split at natural knowledge boundaries
from pymrsf import smart_chunk
pieces = smart_chunk(long_document)   # requires local provider

# Probe what the model already knows
from pymrsf import probe
result = probe("To be or not to be, that is the question.")
print(result["knowledge_score"])  # 92/100 — Shakespeare is memorized
```

---

## Why pymrsf?

Standard RAG retrieves by *relevance* (cosine similarity). A chunk can be highly relevant yet
contain *only facts the model already memorized during training* — wasted context window.

`pymrsf` adds two orthogonal signals:

| Factor | What It Measures | Default Weight |
|--------|-----------------|----------------|
| **Novelty** | New information vs. model's prior knowledge | 40% |
| **Relevance** | Query–chunk semantic similarity | 40% |
| **Query Ignorance** | Does the model not know the answer? | 20% |
| **Diversity** | Dedup: does a better chunk cover this already? | post-filter |

---

## Installation

```bash
# OpenAI API (recommended to start — no 4 GB model download)
pip install pymrsf[openai]

# Anthropic API
pip install pymrsf[anthropic]

# Local model — full features including probing and smart_chunk
pip install pymrsf[local]

# All providers
pip install pymrsf[all]
```

All providers require [Ollama](https://ollama.ai/) for embeddings:

```bash
ollama pull nomic-embed-text
```

---

## Core API

### RAG scoring

```python
from pymrsf import score_chunk, score_chunks, filter_chunks, smart_filter

# Single chunk
result = score_chunk("Backpropagation uses the chain rule.", query="How does backprop work?")
print(result["rag_score"])   # 0–100
print(result["verdict"])     # "excellent" / "good" / "moderate" / "weak" / "skip"

# Many chunks at once
results = score_chunks(chunks, query="...")

# Filter to the best
kept = filter_chunks(chunks, query="...", min_rag_score=50, top_k=5, diversity_threshold=0.85)

# Adaptive budget — returns 0 chunks when the model already knows the answer
smart = smart_filter(chunks, query="...")
print(smart["budget_applied"])   # "high" / "medium" / "low" / "none"
```

### Async support

```python
import asyncio
from pymrsf import filter_chunks_async

useful = asyncio.run(filter_chunks_async(chunks, query="...", min_rag_score=50))
```

### Configurable thresholds and weights

```python
result = score_chunk(
    chunk, query="...",
    weights={"novelty": 0.5, "relevance": 0.3, "query_ignorance": 0.2},
    relevance_cutoff=0.4,
    thresholds=[
        {"label": "excellent", "min": 75},
        {"label": "good",      "min": 55},
        {"label": "weak",      "min": 30},
        {"label": "skip",      "min":  0},
    ],
)
```

### Surprise-guided chunking (local provider)

```python
from pymrsf import smart_chunk

# Chunk boundaries placed at natural knowledge transitions, not fixed sizes
pieces = smart_chunk(long_article, min_chunk_len=200, max_chunk_len=800)
```

### Knowledge probing

```python
from pymrsf import probe, probe_compare

result = probe("The capital of France is Paris.")
print(result["knowledge_score"])   # 95/100

# Compare two phrasings
diff = probe_compare("Paris is in France.", "Paris is the capital of Germany.")
```

---

## Provider comparison

| Feature | Local | OpenAI | Anthropic |
|---------|-------|--------|-----------|
| RAG scoring (novelty + relevance + ignorance) | Full | Approx | Relevance only |
| Knowledge probing | Full | Approx | — |
| `smart_chunk` (surprise-guided) | ✅ | fallback | fallback |
| Async support | ✅ | ✅ | ✅ |
| Score caching | ✅ | ✅ | ✅ |

---

## Runtime configuration

```python
import pymrsf

pymrsf.configure(
    provider="openai",
    embed_timeout=60,
    default_relevance_cutoff=0.4,
)
```

Or via environment variables (`.env` file):

```bash
PYMRSF_PROVIDER=openai
OPENAI_API_KEY=sk-...
PYMRSF_EMBED_MODEL=nomic-embed-text
PYMRSF_EMBED_TIMEOUT=30
```

---

## Experimental: MRSF storage backend

The delta-compression storage system is a research backend — it stores only
"surprise" tokens (40–60% compression) and reconstructs text via O(n) model inference.

```python
from pymrsf.experimental import mrsf_write, mrsf_read, save_index

# Requires local provider
doc = mrsf_write("The Eiffel Tower was built in 1889.")
print(doc["compression"])   # 0.47 — 47% of tokens were predictable

save_index()
results = mrsf_read("famous French landmark", top_k=1)
```

These APIs are stable but may change between minor versions. Import from
`pymrsf.experimental` to signal that dependency clearly; the top-level
re-exports (`from pymrsf import mrsf_write`) continue to work.

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

- [PROVIDER_SUPPORT.md](PROVIDER_SUPPORT.md) — full capability matrix
- [ENV_CONFIG.md](ENV_CONFIG.md) — all environment variables
- [docs/CONCURRENCY.md](docs/CONCURRENCY.md) — threading and process-safety model
- [CHANGELOG.md](CHANGELOG.md) — version history

## License

MIT
