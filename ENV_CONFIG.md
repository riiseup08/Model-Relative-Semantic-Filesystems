# Environment Configuration Reference

pymrsf behavior is configured through environment variables. This document lists all supported variables by provider.

## Common Variables (All Providers)

### `PYMRSF_PROVIDER`
- **Description**: Selects the LLM provider backend
- **Values**: `local` | `openai` | `anthropic`
- **Default**: `local`
- **Example**:
  ```bash
  export PYMRSF_PROVIDER=openai
  ```

### `PYMRSF_MODEL_VERSION`
- **Description**: Model version identifier for cache/storage keys
- **Default**: 
  - local: `mistral-7b-q4km-v1`
  - openai: `gpt-3.5-turbo`
  - anthropic: `claude-3-5-sonnet-20241022`
- **Example**:
  ```bash
  export PYMRSF_MODEL_VERSION=mistral-7b-q4km-v2
  ```

---

## Local Provider Variables

### `PYMRSF_MODEL_PATH`
- **Description**: Path to GGUF model file
- **Required**: Yes (for local provider)
- **Default**: `./models/mistral-7b-v0.1.Q4_K_M.gguf`
- **Example**:
  ```bash
  export PYMRSF_MODEL_PATH=/path/to/model.gguf
  ```

### `PYMRSF_LOGIT_PRECISION`
- **Description**: Decimal precision for logit quantization (reduces memory)
- **Default**: `2`
- **Range**: 0-15
- **Example**:
  ```bash
  export PYMRSF_LOGIT_PRECISION=3
  ```

### `PYMRSF_N_CTX`
- **Description**: Context window size in tokens
- **Default**: `4096`
- **Example**:
  ```bash
  export PYMRSF_N_CTX=8192
  ```

### `PYMRSF_N_GPU_LAYERS`
- **Description**: Number of model layers to offload to GPU
- **Default**: `0` (CPU only)
- **Example**:
  ```bash
  export PYMRSF_N_GPU_LAYERS=32  # For GPU acceleration
  ```

### `PYMRSF_N_THREADS`
- **Description**: Number of CPU threads for inference
- **Default**: `os.cpu_count()` (all available cores)
- **Example**:
  ```bash
  export PYMRSF_N_THREADS=8
  ```

---

## OpenAI Provider Variables

### `OPENAI_API_KEY`
- **Description**: OpenAI API key for authentication
- **Required**: Yes (for openai provider)
- **Default**: None
- **Example**:
  ```bash
  export OPENAI_API_KEY=sk-...
  ```

### `OPENAI_MODEL`
- **Description**: OpenAI model to use
- **Default**: `gpt-3.5-turbo`
- **Example**:
  ```bash
  export OPENAI_MODEL=gpt-4
  ```

---

## Anthropic Provider Variables

### `ANTHROPIC_API_KEY`
- **Description**: Anthropic API key for authentication
- **Required**: Yes (for anthropic provider)
- **Default**: None
- **Example**:
  ```bash
  export ANTHROPIC_API_KEY=sk-ant-...
  ```

---

## Embedding Variables (All Providers)

**Note**: All providers currently require Ollama for embeddings.

### `PYMRSF_OLLAMA_BASE`
- **Description**: Ollama API base URL
- **Default**: `http://localhost:11434`
- **Example**:
  ```bash
  export PYMRSF_OLLAMA_BASE=http://192.168.1.100:11434
  ```

### `PYMRSF_EMBED_MODEL`
- **Description**: Ollama embedding model name
- **Default**: `nomic-embed-text`
- **Example**:
  ```bash
  export PYMRSF_EMBED_MODEL=mxbai-embed-large
  ```

### `PYMRSF_EMBED_TIMEOUT`
- **Description**: Embedding request timeout in seconds
- **Default**: `30`
- **Example**:
  ```bash
  export PYMRSF_EMBED_TIMEOUT=60
  ```

---

## Example .env Files

### Local Provider (Full Features)

```bash
# .env.local
PYMRSF_PROVIDER=local
PYMRSF_MODEL_PATH=./models/mistral-7b-v0.1.Q4_K_M.gguf
PYMRSF_MODEL_VERSION=mistral-7b-q4km-v1
PYMRSF_LOGIT_PRECISION=2
PYMRSF_N_CTX=4096
PYMRSF_N_GPU_LAYERS=0
PYMRSF_N_THREADS=8

# Embeddings (via Ollama)
PYMRSF_OLLAMA_BASE=http://localhost:11434
PYMRSF_EMBED_MODEL=nomic-embed-text
PYMRSF_EMBED_TIMEOUT=30
```

### OpenAI Provider (API-Based)

```bash
# .env.openai
PYMRSF_PROVIDER=openai
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-3.5-turbo
PYMRSF_MODEL_VERSION=gpt-3.5-turbo

# Embeddings (still via Ollama)
PYMRSF_OLLAMA_BASE=http://localhost:11434
PYMRSF_EMBED_MODEL=nomic-embed-text
PYMRSF_EMBED_TIMEOUT=30
```

### Anthropic Provider (API-Based)

```bash
# .env.anthropic
PYMRSF_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-your-key-here
PYMRSF_MODEL_VERSION=claude-3-5-sonnet-20241022

# Embeddings (still via Ollama)
PYMRSF_OLLAMA_BASE=http://localhost:11434
PYMRSF_EMBED_MODEL=nomic-embed-text
PYMRSF_EMBED_TIMEOUT=30
```

### GPU-Accelerated Local Setup

```bash
# .env.local-gpu
PYMRSF_PROVIDER=local
PYMRSF_MODEL_PATH=./models/mistral-7b-v0.1.Q4_K_M.gguf
PYMRSF_N_GPU_LAYERS=32  # Adjust based on VRAM
PYMRSF_N_CTX=8192       # Larger context with GPU
PYMRSF_N_THREADS=4      # Fewer CPU threads when using GPU

PYMRSF_OLLAMA_BASE=http://localhost:11434
PYMRSF_EMBED_MODEL=nomic-embed-text
```

---

## Loading .env Files

### Using python-dotenv

```python
from dotenv import load_dotenv
load_dotenv('.env.local')  # Load before importing pymrsf

import pymrsf
# Now uses environment variables from .env.local
```

### Using direnv

```bash
# Install direnv: https://direnv.net/
echo "dotenv .env.local" > .envrc
direnv allow
```

### Docker Compose

```yaml
# docker-compose.yml
services:
  app:
    image: myapp
    env_file:
      - .env.local
    volumes:
      - ./models:/app/models
```

---

## Runtime Configuration

Some settings can be configured at runtime:

```python
import pymrsf

# Configure caching
pymrsf.cache.configure_cache(
    enabled=True,
    max_size=10000,
    ttl=3600,
    embedding_max_size=5000,
    embedding_ttl=7200
)

# Check current provider
from pymrsf import PROVIDER, MODEL_VERSION, provider_capabilities
print(f"Provider: {PROVIDER}")
print(f"Model: {MODEL_VERSION}")
print(f"Capabilities: {provider_capabilities()}")
```

---

## Validation

To verify your configuration:

```python
from pymrsf import provider_capabilities

caps = provider_capabilities()
print(f"Provider: {caps['provider']}")
print(f"Supports delta compression: {caps['supports_delta']}")
print(f"Supports knowledge probing: {caps['supports_probe']}")

# Try a basic operation
from pymrsf import tokenize, detokenize
ids = tokenize("Hello world")
text = detokenize(ids)
print(f"Tokenization works: {text == 'Hello world'}")
```
