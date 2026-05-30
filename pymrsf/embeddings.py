"""
pymrsf.embeddings — Multi-provider embeddings

All settings are read from pymrsf.Config (seeded from environment variables).
Supports:
  - Local (Ollama)  : nomic-embed-text via Ollama API (default)
  - OpenAI          : text-embedding-ada-002 via OpenAI API
  - Anthropic       : routes to Ollama (Anthropic deprecated their embedding API)
"""

import logging
import os
import threading

import numpy as np
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential


def _cfg():
    """Return the current Config, or None if pymrsf isn't fully initialised."""
    try:
        import pymrsf
        return pymrsf._config
    except (ImportError, AttributeError):
        return None

_logger = logging.getLogger("pymrsf.embeddings")


def _is_retryable(exc: BaseException) -> bool:
    """Retry on connection/timeout errors and HTTP 5xx responses."""
    import requests
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    if isinstance(exc, requests.exceptions.ConnectionError):
        return True
    if isinstance(exc, requests.exceptions.Timeout):
        return True
    if isinstance(exc, requests.exceptions.HTTPError):
        resp = getattr(exc, "response", None)
        return resp is not None and resp.status_code >= 500
    return False


def _log_retry(retry_state) -> None:
    _logger.warning(
        "embed retry %d/%d after %s: %s",
        retry_state.attempt_number,
        3,
        retry_state.outcome.exception().__class__.__name__,
        retry_state.outcome.exception(),
    )

# Embedding dimension — lazily initialised under a lock (Task 2.5)
_embed_dim_cache: int | None = None
_embed_dim_lock = threading.Lock()


# ── Provider implementations ──────────────────────────────────────────────────

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=2),
    retry=retry_if_exception(_is_retryable),
    after=_log_retry,
    reraise=True,
)
def _embed_with_ollama(text: str) -> np.ndarray:
    import requests
    cfg     = _cfg()
    base    = cfg.ollama_base if cfg else os.getenv("PYMRSF_OLLAMA_BASE", "http://localhost:11434")
    model   = cfg.embed_model if cfg else os.getenv("PYMRSF_EMBED_MODEL", "nomic-embed-text")
    timeout = cfg.embed_timeout if cfg else int(os.getenv("PYMRSF_EMBED_TIMEOUT", "30"))
    r = requests.post(
        f"{base}/api/embed",
        json={"model": model, "input": text},
        timeout=timeout,
    )
    r.raise_for_status()
    return np.array(r.json()["embeddings"][0], dtype="float32")


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=2),
    retry=retry_if_exception(_is_retryable),
    after=_log_retry,
    reraise=True,
)
def _embed_with_openai(text: str) -> np.ndarray:
    from openai import OpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OpenAI embeddings require OPENAI_API_KEY environment variable.")
    cfg = _cfg()
    configured = cfg.embed_model if cfg else os.getenv("PYMRSF_EMBED_MODEL", "nomic-embed-text")
    model = configured if configured.startswith("text-embedding") else "text-embedding-ada-002"
    client = OpenAI(api_key=api_key)
    r = client.embeddings.create(input=text, model=model)
    return np.array(r.data[0].embedding, dtype="float32")


# ── Public API ────────────────────────────────────────────────────────────────

def embed(text: str) -> np.ndarray:
    """Generate an embedding vector for text using the configured provider.

    By default (allow_provider_fallback=False) any provider failure raises
    RuntimeError immediately — no silent re-routing.

    Args:
        text: Text to embed.

    Returns:
        np.ndarray: Embedding vector.

    Raises:
        RuntimeError: If the configured provider fails and fallback is disabled.
    """
    cfg = _cfg()
    provider = cfg.provider if cfg else os.getenv("PYMRSF_PROVIDER", "local").lower()
    allow_fallback = cfg.allow_provider_fallback if cfg else (
        os.getenv("PYMRSF_ALLOW_PROVIDER_FALLBACK", "false").lower() == "true"
    )

    if provider == "openai":
        result = _embed_with_openai(text)
    elif provider == "anthropic":
        # Anthropic deprecated their embedding API; route to Ollama
        result = _embed_with_ollama(text)
    else:
        # local provider — use Ollama
        try:
            result = _embed_with_ollama(text)
        except Exception as ollama_exc:
            if allow_fallback and os.getenv("OPENAI_API_KEY"):
                _logger.warning(
                    "PROVIDER FALLBACK: local/ollama → openai. "
                    "Ollama embedding failed (%s). "
                    "Text routed to OpenAI. "
                    "Call pymrsf.configure(allow_provider_fallback=False) to disable.",
                    ollama_exc,
                )
                result = _embed_with_openai(text)
            else:
                raise RuntimeError(
                    f"Ollama embedding failed: {ollama_exc}. "
                    "Ensure Ollama is running with: ollama pull nomic-embed-text\n"
                    "Call pymrsf.configure(allow_provider_fallback=True) to enable "
                    "automatic fallback to OpenAI."
                ) from ollama_exc

    # Validate dimension consistency — benign race on write (same value)
    global _embed_dim_cache
    cached_dim = _embed_dim_cache  # lock-free snapshot
    if cached_dim is None:
        _embed_dim_cache = len(result)
    elif len(result) != cached_dim:
        raise RuntimeError(
            f"Embedding dimension mismatch: expected {cached_dim}, got {len(result)}. "
            "This may happen if the embedding model changed between calls."
        )

    return result


def get_embedding_dim() -> int:
    """Return the embedding dimension for the current model.

    Returns:
        int: Embedding dimension (768 for nomic-embed-text, 1536 for ada-002).
    """
    global _embed_dim_cache
    if _embed_dim_cache is not None:
        return _embed_dim_cache

    try:
        embed("test")  # outside any lock — embed() sets _embed_dim_cache
    except Exception:
        cfg = _cfg()
        provider = cfg.provider if cfg else os.getenv("PYMRSF_PROVIDER", "local").lower()
        with _embed_dim_lock:
            if _embed_dim_cache is None:
                _embed_dim_cache = 1536 if provider == "openai" else 768
    return _embed_dim_cache
