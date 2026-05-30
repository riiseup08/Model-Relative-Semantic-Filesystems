# pymrsf v0.4.1 Baseline

Captured before v0.5 refactor (branch `v0.5-refactor`, tag `v0.4.1-baseline`).

## Test Results

```
66 passed, 3 warnings in ~42s
```

Warnings are harmless SwigPy deprecations from the faiss-cpu wheel (no __module__ attribute on builtin types).

## CLI capabilities

Provider: local (llama-cpp-python). When the local model is not loaded, provider_capabilities() returns supports_logits=False, supports_probe=False with graceful degradation.

## Package state

- `pymrsf/__init__.py` — flat namespace; all symbols exported at top level
- `pymrsf/storage.py` — SQLite + FAISS persistence (research backend)
- `pymrsf/inspect.py` — inspection helpers
- `pymrsf/benchmark.py` — Canterbury corpus benchmarks
- `pymrsf/rag.py` — RAG scoring (headline feature)
- `pymrsf/probe.py` — knowledge probing
- `pymrsf/chunker.py` — surprise-guided chunking
- `pymrsf/cache.py` — score + embedding cache
- `pymrsf/embeddings.py` — multi-provider embeddings
- `pymrsf/core.py` — provider backends, ModelSession
- `pymrsf/cli.py` — CLI entry point

## Known gaps (addressed in v0.5)

- `configure()` does not wire through to modules (modules read os.getenv at call time)
- Score cache uses dict with O(n) eviction scan
- `smart_chunk` re-detokenizes every prefix (O(n²))
- `print()` used throughout instead of `logging`
- No HTTP retry on transient embed failures
- Silent provider fallback in embeddings.py
- Mutable module globals not locked
