# Contributing to pymrsf

Thanks for your interest! This document covers how to set up a development
environment, run tests, and submit changes.

## Dev environment

```bash
# Clone the repo
git clone https://github.com/riiseup08/mrsf.git
cd mrsf

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install with dev dependencies
pip install -e ".[dev]"

# For local model testing (optional, requires a GGUF model):
pip install -e ".[local]"
```

You also need [Ollama](https://ollama.ai/) running locally for embeddings:

```bash
ollama pull nomic-embed-text
```

## Running tests

```bash
# Run the full test suite (no local model required)
pytest tests/ -v

# Run fast tests only
pytest tests/ -v -m "not slow"

# Run with coverage
pytest tests/ --cov=pymrsf

# Run a specific test file
pytest tests/test_core.py -v
```

The test suite is designed to pass **without a local GGUF model**. Tests that
need a local model are skipped automatically.

## Code style

We use [ruff](https://docs.astral.sh/ruff/) for linting and formatting:

```bash
# Lint
ruff check pymrsf/

# Format
ruff format pymrsf/
```

## Commit message convention

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>: <short description>

<optional body>
```

Types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `ci`.

Examples:
```
feat: add OpenAI provider support
fix: correct O(n²) smart_chunk prefix detokenize
docs: add provider matrix to README
test: add hypothesis round-trip property tests
```

## Pull request process

1. Open an issue first to discuss the change you want to make.
2. Create a feature branch from `main`.
3. Write tests for your change.
4. Ensure all existing tests still pass.
5. Run `ruff check` and `ruff format` on your changes.
6. Open a PR against `main` with a clear title and description.

## Project structure

```
pymrsf/
    __init__.py         # Public API exports and Config
    core.py             # Provider backends and lazy model loading
    rag.py              # RAG scoring (score_chunk, filter_chunks)
    chunker.py          # Surprise-guided chunking (smart_chunk)
    probe.py            # Knowledge probing (probe)
    embeddings.py       # Embedding generation
    cache.py            # LRU score and embedding caches
    cli.py              # CLI entry point
    experimental/       # Research-grade MRSF storage backend
        __init__.py
        storage.py      # Delta-compression write/read
        inspect.py      # Delta inspection
        benchmark.py    # Compression/latency benchmarks
tests/
    test_core.py
    test_rag.py
    test_probe.py
    test_cache.py
    test_config.py
    test_async.py
    test_integration.py
    test_storage.py
    test_cache_perf.py
    test_chunker_perf.py
    experimental/
        test_roundtrip.py  # Hypothesis property tests
docs/
    CONCURRENCY.md
    MIGRATION.md
    api.md
    cookbook/
```
