import os
import numpy as np
import requests

# Environment variable configuration
OLLAMA_BASE = os.getenv("PYMRSF_OLLAMA_BASE", "http://localhost:11434")
EMBED_MODEL = os.getenv("PYMRSF_EMBED_MODEL", "nomic-embed-text")
EMBED_TIMEOUT = int(os.getenv("PYMRSF_EMBED_TIMEOUT", "30"))  # seconds

# Cache for embedding dimension (determined at runtime)
_embed_dim_cache = None


def get_embedding_dim() -> int:
    """Get the embedding dimension for the current model.
    
    Returns:
        int: Embedding dimension (typically 768 for nomic-embed-text)
    """
    global _embed_dim_cache
    if _embed_dim_cache is None:
        # Test with a short string to determine dimension
        try:
            test_vec = embed("test")
            _embed_dim_cache = len(test_vec)
        except Exception:
            # Fallback to default
            _embed_dim_cache = 768
    return _embed_dim_cache


def embed(text: str) -> np.ndarray:
    """Generate embedding vector for text using Ollama.
    
    Note: Embeddings currently require Ollama to be running locally,
    even if the main LLM provider is OpenAI or Anthropic.
    
    To configure:
      - PYMRSF_OLLAMA_BASE: Ollama API base URL (default: http://localhost:11434)
      - PYMRSF_EMBED_MODEL: Embedding model name (default: nomic-embed-text)
      - PYMRSF_EMBED_TIMEOUT: Request timeout in seconds (default: 30)
    
    Args:
        text: Text to embed
        
    Returns:
        np.ndarray: Embedding vector (typically 768-dimensional)
        
    Raises:
        RuntimeError: If Ollama is unavailable or request fails
    """
    try:
        r = requests.post(
            f"{OLLAMA_BASE}/api/embed",
            json={"model": EMBED_MODEL, "input": text},
            timeout=EMBED_TIMEOUT
        )
        r.raise_for_status()
        result = np.array(r.json()["embeddings"][0], dtype="float32")
        
        # Validate dimension consistency
        global _embed_dim_cache
        if _embed_dim_cache is None:
            _embed_dim_cache = len(result)
        elif len(result) != _embed_dim_cache:
            raise RuntimeError(
                f"Embedding dimension mismatch: expected {_embed_dim_cache}, got {len(result)}"
            )
        
        return result
    except requests.exceptions.Timeout:
        raise RuntimeError(
            f"Ollama embedding request timed out after {EMBED_TIMEOUT}s. "
            f"Consider increasing PYMRSF_EMBED_TIMEOUT."
        )
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            f"Cannot connect to Ollama at {OLLAMA_BASE}. "
            f"Ensure Ollama is running with: ollama serve\n"
            f"Or set PYMRSF_OLLAMA_BASE to your Ollama instance URL."
        )
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            raise RuntimeError(
                f"Embedding model '{EMBED_MODEL}' not found. "
                f"Pull it with: ollama pull {EMBED_MODEL}\n"
                f"Or set PYMRSF_EMBED_MODEL to an available model."
            )
        else:
            raise RuntimeError(f"Ollama API error: {e}")
    except Exception as e:
        raise RuntimeError(f"Embedding failed: {e}")