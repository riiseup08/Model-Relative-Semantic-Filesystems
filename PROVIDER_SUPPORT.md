# Provider Feature Support Matrix

pymrsf supports three providers with different feature capabilities:

## Feature Matrix

| Feature | local | openai | anthropic |
|---------|-------|--------|-----------|
| **Core Features** |
| Tokenize/Detokenize | ✅ Full | ⚠️ Limited* | ⚠️ Limited* |
| Quantized Argmax | ✅ Yes | ✅ Yes | ❌ No |
| Get Surprises | ✅ Full | ⚠️ Logprob-based | ❌ No |
| Compute Delta | ✅ Yes | ❌ No | ❌ No |
| ModelSession | ✅ Yes | ❌ No | ❌ No |
| Next Token Greedy | ✅ Yes | ❌ No | ❌ No |
| **RAG Features** |
| Score Chunk | ✅ Full scoring | ⚠️ Relevance-only | ⚠️ Relevance-only |
| Knowledge Probing | ✅ Yes | ⚠️ Limited | ❌ No |
| Embedding Support | ✅ Ollama required | ✅ Ollama required | ✅ Ollama required |
| Novelty Detection | ✅ Yes | ❌ No | ❌ No |
| **Storage Features** |
| Delta Compression Write | ✅ Yes | ❌ No | ❌ No |
| Delta Compression Read | ✅ Yes | ❌ No | ❌ No |
| FAISS Indexing | ✅ Yes | ✅ Yes | ✅ Yes |
| **Diagnostic Features** |
| Inspect | ✅ Yes | ❌ No | ❌ No |
| Rebuild Explained | ✅ Yes | ❌ No | ❌ No |
| Benchmark | ✅ Yes | ❌ No | ❌ No |

\* OpenAI and Anthropic tokenization is simulated using tiktoken/approximations, not actual model tokenization.

## Provider-Specific Notes

### Local Provider (`PYMRSF_PROVIDER=local`)
- **Installation**: `pip install pymrsf[local]`
- **Requirements**: 
  - llama-cpp-python
  - Local GGUF model file (e.g., Mistral 7B Q4_K_M)
- **Features**: Full support for all pymrsf features
- **Use Cases**: 
  - Delta compression storage
  - Advanced knowledge probing
  - Offline/privacy-sensitive deployments
  - Research and experimentation

### OpenAI Provider (`PYMRSF_PROVIDER=openai`)
- **Installation**: `pip install pymrsf` (no [local] extra needed)
- **Requirements**: 
  - openai Python package
  - OPENAI_API_KEY environment variable
- **Features**: 
  - Basic surprise detection via logprobs
  - Relevance-only RAG scoring
  - No delta compression
- **Use Cases**: 
  - API-based RAG deployments
  - When local model resources unavailable
  - Hybrid systems (OpenAI for generation, local for storage)

### Anthropic Provider (`PYMRSF_PROVIDER=anthropic`)
- **Installation**: `pip install pymrsf` (no [local] extra needed)
- **Requirements**: 
  - anthropic Python package
  - ANTHROPIC_API_KEY environment variable
- **Features**: 
  - Very limited (embeddings + basic relevance)
  - No surprise detection
  - No delta compression
- **Use Cases**: 
  - Anthropic-centric pipelines
  - Simple relevance-based RAG only

## Programmatic Capability Checks

Use `provider_capabilities()` to check feature support at runtime:

```python
from pymrsf import provider_capabilities

caps = provider_capabilities()

# Check specific capabilities
if caps["supports_logits"]:
    # Can use raw model access features
    pass

if caps["supports_delta"]:
    # Can use delta compression
    from pymrsf import mrsf_write, mrsf_read
    mrsf_write(text)

if caps["supports_probe"]:
    # Can use knowledge probing
    from pymrsf import probe
    result = probe(text)

# Available capability keys:
# - supports_logits: Raw model logits access
# - supports_probe: Knowledge probing via compression
# - supports_delta: Delta compression storage
# - supports_sessions: ModelSession with KV caching
# - supports_true_surprises: Precise surprise detection
# - supports_embeddings: Embedding generation (all providers via Ollama)
# - supports_tokenization: True model tokenization
# - provider: "local" | "openai" | "anthropic"
```

## Error Handling by Provider

All unsupported features raise consistent errors:

```python
# Example: Trying delta compression on OpenAI
from pymrsf import mrsf_write

result = mrsf_write("Some text")
# Returns: {"error": "Delta compression requires local provider", "message": "..."}

# Example: Trying probe on Anthropic
from pymrsf import probe

result = probe("Some text")
# Returns: {"error": "Probe requires local/OpenAI provider", "message": "..."}
```

## Recommended Provider Choice

| Use Case | Recommended Provider |
|----------|---------------------|
| Full features, research, privacy | **local** |
| API-based RAG with limited features | **openai** |
| Simple relevance-only RAG | **openai** or **anthropic** |
| Delta compression storage | **local** only |
| Knowledge probing | **local** or **openai** |
| Offline deployment | **local** only |
| Minimal resource usage | **openai** or **anthropic** |
