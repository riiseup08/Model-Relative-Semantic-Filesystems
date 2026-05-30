"""
pymrsf.embeddings — Multi-provider embeddings

Supports:
  - Local (Ollama)  : nomic-embed-text via Ollama API (default)
  - OpenAI          : text-embedding-ada-002 via OpenAI API
  - Anthropic       : text embeddings via Anthropic API (falls back to Ollama
                      only when PYMRSF_ALLOW_PROVIDER_FALLBACK=true)

Configuration:
  - PYMRSF_OLLAMA_BASE              : Ollama API base URL (default: http://localhost:11434)
  - PYMRSF_EMBED_MODEL              : Embedding model name (default: nomic-embed-text)
  - PYMRSF_EMBED_TIMEOUT            : Request timeout in seconds (default: 30)
  - PYMRSF_PROVIDER                 : Provider used to route embedding strategy
  - PYMRSF_ALLOW_PROVIDER_FALLBACK  : Allow silent fallback to another provider on failure
                                       (default: false — fail-fast for safety)
  - OPENAI_API_KEY                  : Required for OpenAI embeddings
  - ANTHROPIC_API_KEY               : Required for Anthropic embeddings

Production note:
  By default (PYMRSF_ALLOW_PROVIDER_FALLBACK=false), any provider failure raises
  RuntimeError immediately. Set to "true" to enable fallback with a warning log.
"""

import logging
import os
import threading

import numpy as np
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential


def _cfg():
    """Return the current Config, falling back to env vars if pymrsf isn't fully initialised yet."""
    try:
        from pymrsf import _config
        return _config
    except ImportError:
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

# Environment variable configuration — read at call time via get_config() (Task 3.3)
# These module-level defaults remain for backward compat with existing .env setups.
OLLAMA_BASE   = os.getenv("PYMRSF_OLLAMA_BASE",  "http://localhost:11434")
EMBED_MODEL   = os.getenv("PYMRSF_EMBED_MODEL",  "nomic-embed-text")
EMBED_TIMEOUT = int(os.getenv("PYMRSF_EMBED_TIMEOUT", "30"))
PROVIDER      = os.getenv("PYMRSF_PROVIDER", "local").lower()

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
    base    = (cfg.ollama_base   if cfg else None) or os.getenv("PYMRSF_OLLAMA_BASE", OLLAMA_BASE)
    model   = (cfg.embed_model   if cfg else None) or os.getenv("PYMRSF_EMBED_MODEL", EMBED_MODEL)
    timeout = (cfg.embed_timeout if cfg else None) or int(os.getenv("PYMRSF_EMBED_TIMEOUT", str(EMBED_TIMEOUT)))
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
    cfg   = _cfg()
    model = (cfg.embed_model if cfg else None) or os.getenv("PYMRSF_EMBED_MODEL", "text-embedding-ada-002")
    client = OpenAI(api_key=api_key)
    r = client.embeddings.create(input=text, model=model)
    return np.array(r.data[0].embedding, dtype="float32")


def _embed_with_anthropic(text: str) -> np.ndarray:
    """Embed via Anthropic. Falls back to Ollama only if PYMRSF_ALLOW_PROVIDER_FALLBACK=true."""
    allow_fallback = os.getenv("PYMRSF_ALLOW_PROVIDER_FALLBACK", "false").lower() == "true"
    try:
        from anthropic import Anthropic
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("Anthropic embeddings require ANTHROPIC_API_KEY environment variable.")
        client = Anthropic(api_key=api_key)
        r = client.embeddings.create(
            model=os.getenv("PYMRSF_EMBED_MODEL", "claude-3-haiku-20240307"),
            input=text,
        )
        return np.array(r.embedding, dtype="float32")
    except (ImportError, AttributeError) as exc:
        if allow_fallback:
            _logger.warning(
                "PROVIDER FALLBACK: anthropic → ollama. "
                "Anthropic embedding API unavailable (%s). "
                "Text routed to Ollama. Set PYMRSF_ALLOW_PROVIDER_FALLBACK=false to disable.",
                exc,
            )
            return _embed_with_ollama(text)
        raise RuntimeError(
            f"Anthropic embedding API unavailable: {exc}. "
            "Set PYMRSF_ALLOW_PROVIDER_FALLBACK=true to enable automatic fallback to Ollama."
        ) from exc


# ── Public API ────────────────────────────────────────────────────────────────

def embed(text: str) -> np.ndarray:
    """Generate an embedding vector for text using the configured provider.

    By default (PYMRSF_ALLOW_PROVIDER_FALLBACK=false) any provider failure
    raises RuntimeError immediately — no silent re-routing.

    Args:
        text: Text to embed.

    Returns:
        np.ndarray: Embedding vector.

    Raises:
        RuntimeError: If the configured provider fails and fallback is disabled.
    """
    cfg = _cfg()
    provider = ((cfg.provider if cfg else None) or os.getenv("PYMRSF_PROVIDER", PROVIDER)).lower()
    allow_fallback = os.getenv("PYMRSF_ALLOW_PROVIDER_FALLBACK", "false").lower() == "true"

    if provider == "openai":
        result = _embed_with_openai(text)
    elif provider == "anthropic":
        result = _embed_with_anthropic(text)
    else:
        # local provider — use Ollama
        try:
            result = _embed_with_ollama(text)
        except Exception as ollama_exc:
            if allow_fallback and os.getenv("OPENAI_API_KEY"):
                _logger.warning(
                    "PROVIDER FALLBACK: local/ollama → openai. "
                    "Ollama embedding failed (%s). "
                    "Text routed to OpenAI. Set PYMRSF_ALLOW_PROVIDER_FALLBACK=false to disable.",
                    ollama_exc,
                )
                result = _embed_with_openai(text)
            else:
                raise RuntimeError(
                    f"Ollama embedding failed: {ollama_exc}. "
                    "Ensure Ollama is running with: ollama pull nomic-embed-text\n"
                    "Set PYMRSF_ALLOW_PROVIDER_FALLBACK=true to enable automatic fallback to OpenAI."
                ) from ollama_exc

    # Validate dimension consistency — write once under lock (Task 2.5)
    global _embed_dim_cache
    if _embed_dim_cache is None:
        with _embed_dim_lock:
            if _embed_dim_cache is None:  # double-checked
                _embed_dim_cache = len(result)
    elif len(result) != _embed_dim_cache:
        raise RuntimeError(
            f"Embedding dimension mismatch: expected {_embed_dim_cache}, got {len(result)}. "
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

    with _embed_dim_lock:
        if _embed_dim_cache is None:
            try:
                embed("test")
            except Exception:
                provider = os.getenv("PYMRSF_PROVIDER", PROVIDER).lower()
                _embed_dim_cache = 1536 if provider == "openai" else 768
    return _embed_dim_cache
