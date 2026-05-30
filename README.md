# pymrsf — Model-Relative Semantic Filtering

**Novelty-aware RAG chunk scoring using language model surprise signals**

pymrsf helps you build smarter RAG pipelines by leveraging the model's own surprise signals to:
- Score chunks based on **novelty** (does the model already know this?) and **relevance** (does it answer the query?)
- Split documents at **semantic boundaries** rather than arbitrary character limits
- Probe what the model already knows to avoid redundant context
- Filter and deduplicate retrieved chunks for optimal RAG performance

## 🚀 Quick Start

```python
from pymrsf import score_chunk, filter_chunks, smart_chunk, probe

# Score a candidate chunk against a query
result = score_chunk("Neural networks learn by adjusting weights...", 
                     query="How does backpropagation work?")
print(result["rag_score"])  # 0-100
print(result["verdict"])     # excellent / good / moderate / weak / skip

# Filter a list of chunks to the most relevant
chunks = retriever.get(query)
best_chunks = filter_chunks(chunks, query="neural network training", 
                           min_rag_score=50, top_k=5)

# Split text at semantic boundaries (surprise-guided)
pieces = smart_chunk(long_document)

# Probe what the model already knows
knowledge = probe("What is the Eiffel Tower?")
```

**📖 More Examples:**
- **[quickstart.py](quickstart.py)** - Copy-paste ready examples to get started in 5 minutes
- **[examples.py](examples.py)** - Comprehensive examples covering all features

## 📦 Installation

### Local Models (Full Features)
```bash
pip install pymrsf[local]
```
Requires a GGUF model file (e.g., Mistral 7B). Set the path in `.env`:
```
PYMRSF_MODEL_PATH=/path/to/mistral-7b-v0.1.Q4_K_M.gguf
PYMRSF_PROVIDER=local
```

### API-Based (Lightweight)
```bash
# For OpenAI
pip install pymrsf[openai]

# For Anthropic
pip install pymrsf[anthropic]

# For everything
pip install pymrsf[all]
```

Set your API keys in `.env`:
```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-...
PYMRSF_PROVIDER=openai  # or anthropic
```

## 🎯 Key Features

### 1. Novelty-Aware RAG Scoring
Score chunks based on both relevance and novelty:
```python
from pymrsf.rag import score_chunk

result = score_chunk(
    "The Transformer architecture uses self-attention...",
    query="How do transformers work?",
    relevance_cutoff=0.7
)

print(f"RAG Score: {result['rag_score']}/100")
print(f"Novelty: {result['novelty_score']:.2f}")
print(f"Relevance: {result['relevance_score']:.2f}")
print(f"Verdict: {result['verdict']}")
```

### 2. Smart Chunking
Split documents at semantic boundaries instead of arbitrary limits:
```python
from pymrsf import smart_chunk

chunks = smart_chunk(
    long_document,
    min_chunk_len=100,
    max_chunk_len=1000,
    target_chunk_len=500
)
```

### 3. Knowledge Probing
Check what the model already knows:
```python
from pymrsf import probe

result = probe("Paris is the capital of France")
print(f"Model surprise: {result['avg_surprise']:.2f}")
print(f"Known: {result['known']}")  # True if model already knows this
```

### 4. Incremental Filtering
Filter chunks with automatic deduplication:
```python
from pymrsf import filter_chunks

# Automatically removes redundant chunks
best = filter_chunks(
    all_chunks,
    query="machine learning basics",
    min_rag_score=50,
    top_k=5,
    remove_duplicates=True
)
```

## 🔧 Configuration

Create a `.env` file in your project:

```bash
# Provider (local, openai, or anthropic)
PYMRSF_PROVIDER=local

# Local provider settings
PYMRSF_MODEL_PATH=/path/to/model.gguf
PYMRSF_N_CTX=2048
PYMRSF_N_GPU_LAYERS=35

# Embedding settings
PYMRSF_OLLAMA_BASE=http://localhost:11434
PYMRSF_EMBED_MODEL=nomic-embed-text
PYMRSF_EMBED_TIMEOUT=30

# API keys (for OpenAI/Anthropic providers)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-...

# Cache settings
PYMRSF_CACHE_ENABLED=true
PYMRSF_CACHE_MAX_SIZE=10000
PYMRSF_CACHE_TTL=3600
```

## 📊 Feature Support Matrix

| Feature                  | Local | OpenAI | Anthropic |
|-------------------------|-------|--------|-----------|
| tokenize/detokenize     | ✓     | ✓*     | ✓*        |
| embeddings              | ✓     | ✓      | ✓         |
| surprises (logits)      | ✓     | ✓**    | ✗         |
| delta compression       | ✓     | ✗      | ✗         |
| knowledge probing       | ✓     | ✗      | ✗         |
| stateful sessions       | ✓     | ✗      | ✗         |
| raw model access        | ✓     | ✗      | ✗         |

\* Approximations via tiktoken  
\** Limited via API logprobs

### Local-Only Features
- `compute_delta()` - requires exact token prediction
- `ModelSession` - requires KV cache access
- `get_raw_lm()` - requires direct model object
- `mrsf_write()` - requires delta compression
- `probe()` - requires token-level surprises

### Multi-Provider Features
- `score_chunk()` - available everywhere (degrades gracefully)
- `filter_chunks()` - available everywhere
- `embed()` - available everywhere
- Basic RAG scoring works on all providers

## 🛠️ CLI Usage

```bash
# Probe model knowledge
pymrsf probe "The Eiffel Tower is in Paris"

# Score a chunk
pymrsf score "Neural networks learn..." --query "How does ML work?"

# Check provider capabilities
pymrsf capabilities
```

## 📚 Running Examples

```bash
# Run the quick start guide (copy-paste ready examples)
python quickstart.py

# Run all comprehensive examples
python examples.py

# Run a specific example (1-10)
python examples.py 5  # Run example 5: Complete RAG Pipeline
```

Examples included:
1. Basic chunk scoring
2. Filtering and ranking
3. Smart chunking
4. Knowledge probing
5. Complete RAG pipeline
6. Async operations
7. Caching
8. Custom weights
9. Multi-provider setup
10. Production-ready system

## 🧪 Async Support

```python
import asyncio
from pymrsf.rag import score_chunk_async, filter_chunks_async

async def main():
    result = await score_chunk_async("...", query="...")
    useful = await filter_chunks_async(chunks, query, min_rag_score=50)

asyncio.run(main())
```

## 📚 Advanced Usage

### Caching
```python
from pymrsf import cache

# Configure cache
cache.configure_cache(enabled=True, max_size=10000, ttl=3600)

# View cache stats
cache.print_cache_stats()

# Clear cache
cache.clear_cache()
```

### Custom Scoring
```python
result = score_chunk(
    chunk,
    query=query,
    novelty_weight=0.6,      # Weight for novelty score
    relevance_weight=0.4,    # Weight for relevance score
    relevance_cutoff=0.7,    # Minimum relevance threshold
    verbose=True             # Show detailed metrics
)
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

MIT License - see LICENSE file for details

## 🔗 Links

- Documentation: (Coming soon)
- Issues: (GitHub issues)
- PyPI: (Coming soon)

## 💡 How It Works

pymrsf uses the language model's own "surprise" signal (perplexity/logits) to determine:
1. **Novelty**: Does the model already know this information?
2. **Relevance**: Is this information relevant to the query?
3. **Query Ignorance**: Does the model already know the answer?
4. **Incremental Novelty**: Has this already been covered by previous chunks?

This results in more efficient RAG pipelines that only include truly useful context.

## ⚙️ Requirements

- Python 3.10+
- For local models: GGUF model file (4-7GB)
- For API providers: Valid API keys
- For embeddings: Ollama running locally (or API keys)

## 🐛 Troubleshooting

### Local provider issues
```bash
# Check if model file exists
ls -lh /path/to/model.gguf

# Verify llama-cpp-python installation
pip show llama-cpp-python
```

### Embedding issues
```bash
# Check Ollama is running
curl http://localhost:11434/api/tags

# Pull embedding model
ollama pull nomic-embed-text
```

### API provider issues
```bash
# Verify API keys are set
echo $OPENAI_API_KEY
echo $ANTHROPIC_API_KEY
```

---

Made with ❤️ for better RAG pipelines
