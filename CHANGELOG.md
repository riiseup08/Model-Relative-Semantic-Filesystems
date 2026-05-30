# Changelog

All notable changes to pymrsf will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.0] — 2026-05-10

### Breaking changes
- `pymrsf.storage`, `pymrsf.inspect`, `pymrsf.benchmark` moved to `pymrsf.experimental`.
  Top-level re-exports remain for backward compat and will be removed in v0.6.
- `rebuild_faiss_from_sqlite()` now emits `DeprecationWarning`. Use `reset_index_metadata()`.
- Default embed behavior is **fail-fast** on provider failure. Set
  `PYMRSF_ALLOW_PROVIDER_FALLBACK=true` to restore silent fallback (now logged as WARNING).

### New features
- `pymrsf.experimental` subpackage — MRSF storage clearly scoped as research-grade.
- `pymrsf.configure_logging(level)` — opt-in to library log output; import is now silent.
- `pymrsf.configure(embed_timeout=…, embed_model=…)` now takes effect at the next call.
- HTTP retry on transient embed failures: 3 attempts, exponential backoff (tenacity).
- `docs/CONCURRENCY.md` — threading model, WAL mode, multi-process patterns documented.

### Fixes
- Score cache: O(1) LRU eviction via `OrderedDict` (was O(n) min-timestamp scan).
- Score cache: `deepcopy` moved outside lock — callers no longer serialize on large objects.
- Score cache: removed unbounded `@lru_cache` on `_cached_text_hash`.
- `_embed_dim_cache` initialised under `threading.Lock` (double-checked locking).
- `smart_chunk` offset recovery: O(n) incremental walk (was O(n²) prefix re-detokenize).
- All `print()` calls converted to `logging` — `import pymrsf` is now side-effect free.

### Tests
- 83 tests, 0 failures (up from 66).
- Hypothesis round-trip property tests for MRSF storage (50 examples; 1000-example slow variant).
- Cache perf + concurrency tests: insertion <100 µs, 8-thread mixed get/set, deepcopy isolation.
- Config live-wiring tests, chunker O(n) perf test.

### Dependencies added
- `tenacity>=8.0.0` (core), `hypothesis>=6.0` (dev)
- `llama-cpp-python` tightened to `>=0.2.50,<0.3.0`

---

## [0.4.0] - 2026-05-07

### Added
- **OpenAI API support** — Use GPT-3.5/GPT-4 without downloading a local model
- **Anthropic API support** — Use Claude models for basic RAG scoring
- **Async support** — Non-blocking `score_chunk_async()`, `score_chunks_async()`, and `filter_chunks_async()`
  - Concurrent scoring with configurable `max_concurrent` parameter
  - 3-5x faster with API providers
  - Ideal for production RAG pipelines
- **Caching system** — Automatic caching of chunk scores and embeddings
  - Configurable LRU cache with TTL support
  - Thread-safe implementation
  - Near-instant scoring for repeated chunks (~100x speedup)
  - Cache statistics and monitoring via `cache.print_cache_stats()`
- Optional dependencies system — Install only what you need:
  - `pymrsf[local]` — Local model support (llama-cpp-python)
  - `pymrsf[openai]` — OpenAI API support
  - `pymrsf[anthropic]` — Anthropic API support
  - `pymrsf[all]` — Everything
- `.env.example` configuration file with examples for all providers
- `example_openai.py` — Complete example using OpenAI provider
- `example_performance.py` — Async and caching benchmarks
- Provider comparison table in README
- Migration guide for upgrading from v0.3

### Changed
- **Breaking (optional dependencies):** `llama-cpp-python` is now optional, not required
  - Existing users: Run `pip install pymrsf[local]` to restore full functionality
  - Default behavior unchanged if llama-cpp-python is already installed
- `score_chunk()` now includes `use_cache` parameter and returns `cached` field in results
- Improved error messages throughout the codebase
- Updated README with API-based quick start (30 seconds to get started!)
- Core module docstring updated with provider information
- Embedding calls now use caching automatically

### Fixed
- Better handling of missing dependencies with helpful error messages
- Clear distinction between features available in each provider

### Performance Improvements
- **3-5x faster** with async scoring on API providers
- **~100x faster** for cached chunks (repeated queries)
- **Up to 10x throughput** improvement with async + caching combined
- Automatic embedding caching reduces redundant API calls

### Limitations by Provider
- **Local:** Full feature support (RAG scoring, knowledge probing, delta compression)
- **OpenAI:** Basic RAG scoring with logprobs API (no full knowledge probing)
- **Anthropic:** Relevance-only RAG scoring (no logprobs available from API)

## [0.3.0] and earlier

See git history for changes prior to v0.4.0.
