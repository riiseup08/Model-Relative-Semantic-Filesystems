# pymrsf — Novelty-Aware RAG Chunk Scoring

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-33%20passing-brightgreen)]()

**Stop wasting context window on information your LLM already knows.**

`pymrsf` scores RAG chunks by measuring **information gain** — not just relevance. It uses the model's own predictive surprise to detect which chunks contain genuinely new information.

**🚀 New in v0.4:** Lightweight API providers (OpenAI, Anthropic) — get started in 30 seconds without downloading a 4GB model!

## The Problem

Standard RAG retrieves chunks by *relevance* (cosine similarity). But a chunk can be highly relevant while containing *only facts the model already memorized during training*. You waste precious context window on redundant information.

Also, if the LLM already *knows the answer* to the query, even novel chunks are less useful. And if two chunks say the same thing, you don't need both.

## The Solution

`pymrsf` introduces **multi-factor novelty-aware scoring**:

| Factor | What It Measures | Weight |
|--------|-----------------|--------|
| **Novelty** | How much *new* information does this chunk contain? | 40% |
| **Relevance** | How related is this chunk to the query? | 40% |
| **Query Ignorance** | Does the model *not* know the answer to your question? | 20% |
| **Diversity** | Does a better chunk already cover this content? | Dedup |

## Quick Start

### 🚀 Get Started in 30 Seconds (No 4GB Model Download!)

The easiest way to try `pymrsf` is with an API provider:

```bash
# Install with OpenAI support (no heavy dependencies!)
pip install pymrsf[openai]

# Set your API key
export OPENAI_API_KEY='sk-...'
```

```python
import os
os.environ["PYMRSF_PROVIDER"] = "openai"  # Use OpenAI instead of local model

from pymrsf.rag import filter_chunks

chunks = [
    "Backpropagation computes gradients using the chain rule.",
    "Neural networks are inspired by the human brain.",
    "The sky is blue because of Rayleigh scattering.",
]

query = "How does backpropagation work?"
useful = filter_chunks(chunks, query, min_rag_score=50, verbose=True)
# → Returns only relevant chunks, saves your context window!
```

### 🏠 Local Model (Advanced Features)

For full features including **knowledge probing** and **delta compression**, use a local model:

```bash
# Install with local model support
pip install pymrsf[local]

# Download a model (one-time, ~4GB)
# Example: Mistral 7B Q4 from https://huggingface.co/TheBloke/Mistral-7B-v0.1-GGUF
```

```python
import os
os.environ["PYMRSF_PROVIDER"] = "local"  # default
os.environ["PYMRSF_MODEL_PATH"] = "./models/mistral-7b-v0.1.Q4_K_M.gguf"

from pymrsf.rag import filter_chunks
from pymrsf import probe

# RAG scoring (same as API mode)
useful = filter_chunks(chunks, query, min_rag_score=50)

# Knowledge probing (local only)
result = probe("To be or not to be, that is the question.")
print(f"Knowledge: {result['knowledge_score']}/100")  # 92/100 (memorized)
```

## Installation Options

Choose based on your needs:

```bash
# Lightweight — OpenAI API (recommended for getting started)
pip install pymrsf[openai]

# Lightweight — Anthropic API  
pip install pymrsf[anthropic]

# Full features — Local model (4GB+ model download required)
pip install pymrsf[local]

# Everything — All providers
pip install pymrsf[all]

# Development
git clone https://github.com/riiseup08/mrsf.git
cd mrsf
pip install -e .[all]
```

### Provider Comparison

| Feature | Local | OpenAI | Anthropic |
|---------|-------|--------|-----------|
| RAG Chunk Scoring | ✅ Full | ⚠️ Relevance-only | ⚠️ Relevance-only |
| Knowledge Probing | ✅ Full | ⚠️ Limited | ❌ |
| Delta Compression | ✅ | ❌ | ❌ |
| Async Support | ✅ | ✅ | ✅ |
| Caching | ✅ | ✅ | ✅ |
| Setup Difficulty | Hard | Easy | Easy |
| Cost | Free | $$ | $$ |
| Privacy | Private | API | API |

**⚠️ Important Notes:**
- **Embeddings**: All providers currently require [Ollama](https://ollama.ai/) running locally for embedding generation (uses `nomic-embed-text` by default)
- **API Approximations**: OpenAI/Anthropic providers use logprob-based approximations for surprise detection. Results are less precise than local models but still effective for RAG scoring.
- See **[PROVIDER_SUPPORT.md](PROVIDER_SUPPORT.md)** for the complete feature matrix and capability details.
- See **[ENV_CONFIG.md](ENV_CONFIG.md)** for all environment variable configuration options.

## Features

### 🎯 RAG Chunk Scoring (Core Feature)

```python
from pymrsf.rag import score_chunk, score_chunks, score_chunks_batch

# Single chunk scoring
result = score_chunk(
    "Backpropagation computes gradients using the chain rule.",
    query="How does backpropagation work?",
    verbose=True
)
print(result["rag_score"])    # 72/100
print(result["verdict"])      # "good"
print(result["query_knowledge"])  # how much model knows the query

# Batch scoring (3-5x faster for many chunks)
results = score_chunks_batch(chunks, query)

# Custom weights (adjust the formula)
weights = {"novelty": 0.5, "relevance": 0.3, "query_ignorance": 0.2}
result = score_chunk(chunk, query, weights=weights)
```

### 🔍 Knowledge Probing (Local/OpenAI Only)

Test how well the model knows specific information:

```python
from pymrsf import probe, provider_capabilities

# Check if probing is available
caps = provider_capabilities()
if caps["supports_probe"]:
    result = probe("To be or not to be, that is the question.")
    print(f"Knowledge: {result['knowledge_score']}/100 ({result['label']})")
    # → Knowledge: 92/100 (memorized) — Shakespeare is well-known
    
    result = probe("My proprietary algorithm uses a novel attention mechanism.")
    print(f"Knowledge: {result['knowledge_score']}/100 ({result['label']})")
    # → Knowledge: 15/100 (unknown) — novel content!
    
    # Get detailed token-level analysis
    print(f"Surprises: {result['surprises']}")  # Which tokens were unpredictable
else:
    print(f"Probing not available with {caps['provider']} provider")
```

**Provider Support:**
- ✅ **Local**: Full precision surprise detection
- ⚠️ **OpenAI**: Approximate via logprobs (less precise but functional)
- ❌ **Anthropic**: Not supported

### 🔧 RAG Pipeline Filter (All Providers)

Complete pipeline: retrieve chunks, score them, filter to the best ones:

```python
from pymrsf.rag import filter_chunks

# Your existing retriever (FAISS, Pinecone, etc.)
chunks = retriever.get(query, top_k=20)

# Filter to only useful chunks
good = filter_chunks(
    chunks,
    query,
    min_rag_score=50,         # skip low-value chunks
    top_k=5,                  # limit context window usage
    diversity_threshold=0.85, # dedup similar chunks
    verbose=True,
)

# Use filtered chunks in your LLM prompt
answer = llm.complete(query, context=good)

# Save context window tokens!
print(f"Reduced {len(chunks)} chunks to {len(good)} high-value chunks")
```

**Works with all providers** - automatically adapts to available features.

### ⚡ Performance Features (NEW in v0.4)

**Async Support** — Non-blocking scoring for production RAG pipelines:

```python
import asyncio
from pymrsf.rag import score_chunk_async, filter_chunks_async

async def my_rag_pipeline(chunks, query):
    # Score chunks without blocking
    useful = await filter_chunks_async(
        chunks,
        query,
        min_rag_score=50,
        max_concurrent=10,  # Score 10 chunks at once
    )
    return useful

# Run it
useful = asyncio.run(my_rag_pipeline(chunks, query))
```

**Caching** — Avoid re-scoring the same chunks:

```python
from pymrsf import cache

# Configure cache (do this once at startup)
cache.configure_cache(
    enabled=True,
    max_size=10000,  # Store up to 10k scored chunks
    ttl=3600,        # Cache for 1 hour
)

# Score chunks (caching happens automatically)
result = score_chunk(chunk, query=query)

# Check cache performance
cache.print_cache_stats()
# → Hit rate: 85.3% (cache is working!)

# Clear cache if needed
cache.clear_cache()
```

**Performance Benefits:**
- 🚀 **Async**: 3-5x faster with API providers (OpenAI, Anthropic)
- 💾 **Caching**: Near-inStorage (Local Only, Experimental)

Store documents efficiently by saving only "surprise" tokens (40-60% compression on average):

```python
from pymrsf import mrsf_write, mrsf_read, save_index, load_index, provider_capabilities

# Check if delta compression is available
caps = provider_capabilities()
if caps["supports_delta"]:
    # Store documents with semantic search + delta compression
    doc1 = mrsf_write("The Eiffel Tower was built in 1889 for the World's Fair.")
    doc2 = mrsf_write("Neural networks learn by adjusting weights through backprop.")
    doc3 = mrsf_write("Python is a popular programming language.")
    
    print(f"Doc1: {doc1['compression']:.1%} compression")
    # → 45.2% compression
    
    # Save FAISS index to disk
    save_index()
    
    # Later: Load index and search
    load_index()
    results = mrsf_read("tell me about the Eiffel Tower", top_k=1)
   Complete Examples

### Quick Start: OpenAI Provider (Recommended for First-Time Users)

```python
import os
os.environ["PYMRSF_PROVIDER"] = "openai"
os.environ["OPENAI_API_KEY"] = "sk-..."
Quick Setup

Choose your provider and create a `.env` file:

**OpenAI (Easiest):**
```bash
PYMRSF_PROVIDER=openai
OPENAI_API_KEY=sk-...

# Embeddings (required for all providers)
PYMRSF_OLLAMA_BASE=http://localhost:11434  # default
PYMRSF_EMBED_MODEL=nomic-embed-text        # default
```

**Anthropic:**
```bash
PYMRSF_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...

# Embeddings (required)
PYMRSF_OLLAMA_BASE=http://localhost:11434
PYMRSF_EMBED_MODEL=nomic-embed-text
```

**Local (Full Features):**
```bash
PYMRSF_PROVIDER=local  # default
PYMRSF_MODEL_PATH=./models/mistral-7b-v0.1.Q4_K_M.gguf
PYMRSF_N_GPU_LAYERS=0  # set to 20-32 for GPU acceleration
PYMRSF_N_CTX=4096

# Embeddings (required)
PYMRSF_OLLAMA_BASE=http://localhost:11434
PYMRSF_EMBED_MODEL=nomic-embed-text
```

### Embeddings Setup (Required for All Providers)

**All providers require Ollama for embeddings:**

1. Install Ollama: https://ollama.ai/
2. Pull the embedding model:
   ```bash
   ollama pull nomic-embed-text
   ```
3. Ollama will run on `http://localhost:11434` by default

**Alternative embedding models:**
- `mxbai-embed-large` - Higher quality, slower
- `all-minilm` - Faster, lower quality

Set via `PYMRSF_EMBED_MODEL` environment variable.

### Where to Get Local Models

- [Mistral 7B GGUF](https://huggingface.co/TheBloke/Mistral-7B-v0.1-GGUF) (recommended, ~4GB)
- [Llama 2 7B GGUF](https://huggingface.co/TheBloke/Llama-2-7B-GGUF)
- Any GGUF model from [TheBloke on Hugging Face](https://huggingface.co/TheBloke)

### Complete Configuration Reference

See **[ENV_CONFIG.md](ENV_CONFIG.md)** for all 15+ environment variables and example `.env` files for each provider.
from pymrsf.rag import score_chunks
How It Works

### RAG Score Formula
```
rag_score = novelty × 0.40 + relevance × 0.40 + query_ignorance × 0.20
```

**Components:**
- **Novelty**: How much new information does the chunk contain? (Measured via compression rate - unpredictable tokens indicate novelty)
- **Relevance**: How related is the chunk to the query? (Cosine similarity between embeddings)
- **Query Ignorance**: Does the model not know the answer? (Probing query compression - if model already knows the answer, chunks are less useful)

**Custom Weights:**
```python
weights = {"novelty": 0.5, "relevance": 0.3, "query_ignorance": 0.2}
result = score_chunk(chunk, query, weights=weights)
```

### Score Interpretation

| Score | Verdict | Action |
|-------|---------|--------|
| 80-100 | Excellent | Prioritize this chunk |
| 60-79 | Good | Include in context |
| 40-59 | Moderate | Include if space allows |
| 20-39 | Weak | Skip if better chunks exist |
| 0-19 | Skip | Model already knows this |

### Provider-Specific Behavior

**Local Provider:**
- Full novelty detection via token-level surprise
- Precise knowledge probing
- Best results, highest computational cost

**OpenAI Provider:**
- Novelty approximated via logprobs (less precise but effective)
- Limited knowledge probing (logprob-based)
- Fast, API-based, good for production

**Anthropic Provider:**
- Relevance-only scoring (no novelty detection)
- No knowledge probing
- Basic but functional for simple RAG pipelines
from pymrsf import probe, provider_capabilities
from pymrsf.rag import score_chunk, filter_chunks
from pymrsf import mrsf_write, mrsf_read, save_index

# Check available features
caps = provider_capabilities()
print(f"Provider: {caps['provider']}")
print(f"Supports probing: {caps['supports_probe']}")
print(f"Supports delta compression: {caps['supports_delta']}")

# Knowledge probing
result = probe("To be or not to be")
print(f"Model knows this {result['knowledge_score']}/100")

# RAdditional Documentation

- **[PROVIDER_SUPPORT.md](PROVIDER_SUPPORT.md)** - Complete feature matrix by provider, capability checks, recommended use cases
- **[ENV_CONFIG.md](ENV_CONFIG.md)** - All environment variables, example `.env` files for each provider
- **[PERFORMANCE.md](PERFORMANCE.md)** - Benchmarks, optimization tips, async best practices
- **[CHANGELOG.md](CHANGELOG.md)** - Version history and migration guides

## Project Status

**v0.4 - Beta** — Core RAG novelty scoring is stable and production-ready. Multi-provider support is fully functional. Delta compression storage is experimental (local provider only).

**What's stable:**
- ✅ RAG chunk scoring (all providers)
- ✅ Knowledge probing (local/OpenAI)
- ✅ Async support (all providers)
- ✅ Caching system

**What's experimental:**
- ⚠️ Delta compression storage (works but API may change)
- ⚠️ Anthropic provider (basic functionality only)

# Delta compression storage
doc = mrsf_write("The Eiffel Tower was built in 1889.")
print(f"Compression: {doc['compression']:.1%}")
save_index()

# Semantic retrieval
results = mrsf_read("famous French landmark", top_k=1)
print(results[0])  # Reconstructed text
```
    print("Delta compression requires local provider")
    print("Install with: pip install pymrsf[local]")
```

**How it works:**
1. Text is tokenized and fed to the model
2. Only "surprising" tokens (where prediction ≠ actual) are stored
3. Reconstruction uses the model to predict missing tokens
4. FAISS provides semantic search over embedded documents

**Use cases:** Cold storage, archival systems, research datasets

**Performance:** ~40-60% compression on average, O(n) reconstruction timepython
from pymrsf import mrsf_write, mrsf_read, save_index

# Write (stores only surprise tokens = ~40% compression)
mrsf_write("The Eiffel Tower is in Paris.")
save_index()

# Read (reconstructs from delta + model)
results = mrsf_read("famous landmark in France")
```

## Examples

See [example_openai.py](example_openai.py) for a complete example using the OpenAI provider.

For local model usage, see the examples in the Features section below.

## Configuration

### Using API Providers (Recommended for Getting Started)

Create a `.env` file (or copy from `.env.example`):

**OpenAI:**
```bash
# .env file
PYMRSF_PROVIDER=openai
OPENAI_API_KEY=sk-...
PYMRSF_MODEL_VERSION=gpt-3.5-turbo  # or gpt-4, gpt-4o
```

**Anthropic:**
```bash
# .env file
PYMRSF_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
PYMRSF_MODEL_VERSION=claude-3-5-sonnet-20241022  # or other Claude models
```

### Using Local Models (Advanced)

```bash
# .env file
PYMRSF_PROVIDER=local  # default
PYMRSF_MODEL_PATH=./models/mistral-7b-v0.1.Q4_K_M.gguf
PYMRSF_N_GPU_LAYERS=0  # set to 20-30 if you have GPU
PYMRSF_N_CTX=4096  # context window size
```

**Where to get local models:**
- [Mistral 7B GGUF](https://huggingface.co/TheBloke/Mistral-7B-v0.1-GGUF) (recommended)
- [Llama 2 7B GGUF](https://huggingface.co/TheBloke/Llama-2-7B-GGUF)
- Any GGUF model from [TheBloke on Hugging Face](https://huggingface.co/TheBloke)

## Scoring Concepts

### RAG Score Formula
```
rag_score = novelty × 0.40 + relevance × 0.40 + query_ignorance × 0.20
```

### What the Scores Mean

| Score | Verdict | Action |
|-------|---------|--------|
| 80-100 | Excellent | Prioritize this chunk |
| 60-79 | Good | Include in context |
| 40-59 | Moderate | Include if space allows |
| 20-39 | Weak | Skip if better chunks exist |
| 0-19 | Skip | Model already knows this |

## Project Structure

```
pymrsf/
├── __init__.py     # Public API exports
├── core.py         # Multi-provider backend (local, OpenAI, Anthropic) with lazy loading
├── embeddings.py   # Ollama embedding API client
├── probe.py        # Knowledge probing (local provider only)
├── rag.py          # RAG chunk scoring with novelty + relevance + diversity
├── storage.py      # Delta compression storage (local provider only, experimental)
├── inspect.py      # Token-level visualization tools
└── benchmark.py    # Compression/latency benchmarks
```

## Upgrading from v0.3

If you're upgrading from an earlier version, here's what changed in v0.4:

**Dependencies are now optional!**
```bash
# Old installation (still works, but heavy)
pip install -e .

# New installation (choose what you need)
pip install -e .[local]    # for local models
pip install -e .[openai]   # for OpenAI API
pip install -e .[all]      # everything
```

**Your existing code still works:**
- Default provider is still `local` (no breaking changes)
- If you have `llama-cpp-python` installed, everything works as before
- No code changes needed unless you want to use API providers

**To use API providers:**
```python
import os
os.environ["PYMRSF_PROVIDER"] = "openai"  # or "anthropic"
# Rest of your code stays the same!
```

## Project Status

**Alpha** — The RAG novelty scoring works and solves a real problem. The delta compression/storage system is experimental.

## License

MIT
