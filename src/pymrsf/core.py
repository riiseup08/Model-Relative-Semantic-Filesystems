"""
pymrsf — multi-provider backend
Supports:
  - local    : any GGUF model via llama-cpp-python (full feature support)
  - openai   : GPT-3.5, GPT-4 via OpenAI API (basic RAG scoring only)
  - anthropic: Claude via Anthropic API (basic RAG scoring only)

Set provider in .env:
  PYMRSF_PROVIDER=local      # default (requires local model)
  PYMRSF_PROVIDER=openai     # lightweight, API-based
  PYMRSF_PROVIDER=anthropic  # lightweight, API-based

Installation:
  pip install pymrsf[local]     # for local models
  pip install pymrsf[openai]    # for OpenAI API
  pip install pymrsf[anthropic] # for Anthropic API
  pip install pymrsf[all]       # everything

Feature Support Matrix:
┌─────────────────────────┬───────┬─────────┬────────────┐
│ Feature                 │ Local │ OpenAI  │ Anthropic  │
├─────────────────────────┼───────┼─────────┼────────────┤
│ tokenize/detokenize     │   ✓   │   ✓*    │     ✓*     │
│ embeddings              │   ✓   │   ✓     │     ✓      │
│ surprises (logits)      │   ✓   │   ✓**   │     ✗      │
│ delta compression       │   ✓   │   ✗     │     ✗      │
│ knowledge probing       │   ✓   │   ✗     │     ✗      │
│ stateful sessions       │   ✓   │   ✗     │     ✗      │
│ raw model access        │   ✓   │   ✗     │     ✗      │
└─────────────────────────┴───────┴─────────┴────────────┘

  * Approximations via tiktoken (not true tokenization)
 ** Limited via API logprobs (threshold-based, not argmax)

Local-Only Functions:
  - compute_delta()     : requires exact token prediction
  - ModelSession        : requires KV cache access
  - get_raw_lm()        : requires direct model object
  - mrsf_write()        : requires delta compression
  - mrsf_inspect()      : requires raw logits
  - probe()             : requires token-level surprises

Multi-Provider Functions:
  - tokenize()          : available everywhere (may be approximate)
  - detokenize()        : available everywhere (may be approximate)
  - embed()             : available everywhere
  - score_chunk()       : available everywhere (degrades gracefully)
  - filter_chunks()     : available everywhere (degrades gracefully)

Note: Advanced features (knowledge probing, compression) require local provider.
"""

import logging
import os
import threading
from dotenv import load_dotenv

_logger = logging.getLogger("pymrsf.core")

load_dotenv()

PROVIDER        = os.getenv("PYMRSF_PROVIDER", "local").lower()
LOGIT_PRECISION = int(os.getenv("PYMRSF_LOGIT_PRECISION", "6"))

# ── Lazy model loading ────────────────────────────────────────────────────────
# The LLM model is NOT loaded at import time. It's loaded on first use.
# This avoids loading a 4GB+ model when you only import for RAG scoring.

_lm = None
_lm_loaded = False

# The local llama_cpp model holds a single mutable KV cache and is NOT reentrant.
# Any code that calls _lm.reset()/eval()/reads _lm.scores must hold this lock so
# concurrent callers (e.g. the threaded async scoring path) can't corrupt each
# other's model state. probe.py acquires it via model_lock().
_model_lock = threading.Lock()


def model_lock() -> threading.Lock:
    """Return the global lock guarding the single local model's mutable state."""
    return _model_lock


def _ensure_model():
    """Lazy-load the LLM model on first actual use."""
    global _lm, _lm_loaded
    if _lm_loaded:
        return
    if PROVIDER == "local":
        try:
            import numpy as np
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "\n[pymrsf] Local provider requires llama-cpp-python.\n"
                "  Install with: pip install pymrsf[local]\n"
                "  Or use a lightweight API provider instead:\n"
                "    Set PYMRSF_PROVIDER=openai (requires OpenAI API key)\n"
                "    Set PYMRSF_PROVIDER=anthropic (requires Anthropic API key)\n"
            )

        GGUF_PATH     = os.getenv("PYMRSF_MODEL_PATH",    "./models/mistral-7b-v0.1.Q4_K_M.gguf")
        N_CTX         = int(os.getenv("PYMRSF_N_CTX",         "4096"))
        N_GPU_LAYERS  = int(os.getenv("PYMRSF_N_GPU_LAYERS",  "0"))
        N_THREADS     = int(os.getenv("PYMRSF_N_THREADS",     str(os.cpu_count())))

        if not os.path.exists(GGUF_PATH):
            raise FileNotFoundError(
                f"\n[pymrsf] Model not found: {GGUF_PATH}\n"
                f"  Set PYMRSF_MODEL_PATH in your .env"
            )

        _logger.info("Loading local model: %s", GGUF_PATH)
        _lm = Llama(
            model_path   = GGUF_PATH,
            n_ctx        = N_CTX,
            n_gpu_layers = N_GPU_LAYERS,
            logits_all   = True,
            verbose      = False,
            n_threads    = N_THREADS,
        )
        _logger.info("Model loaded.")
        _lm_loaded = True
    elif PROVIDER == "openai":
        _lm_loaded = True  # No LLM to load, just mark as ready
    elif PROVIDER == "anthropic":
        _lm_loaded = True  # No LLM to load — Anthropic doesn't expose logprobs
    else:
        raise ValueError(f"[pymrsf] Unknown provider: '{PROVIDER}'")


# ── Local provider ─────────────────────────────────────────────────────────────

def _load_local_backend():
    """Dynamically load the local LLM backend functions."""
    import numpy as np

    _ensure_model()

    def tokenize(text: str) -> list:
        return _lm.tokenize(text.encode("utf-8"), add_bos=True)

    def detokenize(ids: list) -> str:
        """Convert token IDs back to string. Preserves all spaces."""
        return _lm.detokenize(ids).decode("utf-8", errors="replace")

    def _quantized_argmax(raw_logits) -> int:
        q = np.round(np.array(raw_logits, dtype=np.float64), decimals=LOGIT_PRECISION)
        return int(np.argmax(q))

    quantized_argmax = _quantized_argmax

    def get_surprises(text: str) -> tuple:
        """Returns (surprises, heatmap, token_count)"""
        token_ids = tokenize(text)
        n = len(token_ids)

        surprises = []
        heatmap = []

        with _model_lock:
            _lm.reset()
            _lm.eval(token_ids)
            for i in range(n - 1):
                pred_id   = _quantized_argmax(_lm.scores[i])
                actual_id = token_ids[i + 1]
                token_str = detokenize([actual_id]).strip() or f"<{actual_id}>"
                surprised = pred_id != actual_id

                if surprised:
                    surprises.append((i + 1, token_str))

                heatmap.append({
                    "token": token_str,
                    "surprised": surprised,
                    "position": i + 1,
                })

        return surprises, heatmap, n

    def compute_delta(text_or_ids) -> list:
        """Compute delta (surprise positions and token IDs).

        Args:
            text_or_ids: Either a string or a list of token IDs

        Returns:
            List of (position, token_id) tuples for surprise tokens
        """
        if isinstance(text_or_ids, str):
            ids = tokenize(text_or_ids)
        else:
            ids = text_or_ids
        n = len(ids)
        delta = []
        with _model_lock:
            _lm.reset()
            _lm.eval(ids)
            for i in range(n - 1):
                pred   = _quantized_argmax(np.array(_lm.scores[i]))
                actual = ids[i + 1]
                if pred != actual:
                    delta.append((i + 1, actual))
        return delta

    # --- Stateful session for O(n) reconstruction ---
    class ModelSession:
        """
        Maintains a single model state for incremental generation.
        Feed tokens one by one; predict next token from current state.
        """
        def __init__(self):
            _ensure_model()
            self.lm = _lm
            self.reset()

        def reset(self):
            """Reset model state (clear KV cache)."""
            self.lm.reset()
            self._last_logits = None

        def feed(self, token_id: int):
            """Feed a single token to the model and update internal logits."""
            with _model_lock:
                self.lm.eval([token_id])
                if len(self.lm.scores) == 0:
                    self._last_logits = None
                else:
                    self._last_logits = self.lm.scores[-1]

        def predict_next(self) -> int:
            """Return the greedy next token based on current state."""
            if self._last_logits is None:
                raise RuntimeError("No logits available. Call feed() first.")
            return _quantized_argmax(np.array(self._last_logits))

    # Legacy O(n²) version
    def next_token_greedy(context_ids: list) -> int:
        """Legacy O(n²) version – use ModelSession instead."""
        with _model_lock:
            _lm.reset()
            _lm.eval(context_ids)
            return _quantized_argmax(np.array(_lm.scores[len(context_ids) - 1]))

    return {
        "tokenize": tokenize,
        "detokenize": detokenize,
        "quantized_argmax": quantized_argmax,
        "get_surprises": get_surprises,
        "compute_delta": compute_delta,
        "ModelSession": ModelSession,
        "next_token_greedy": next_token_greedy,
        "lm": _lm,
    }


# ── OpenAI provider ────────────────────────────────────────────────────────────

def _load_openai_backend():
    """Dynamically load the OpenAI backend functions."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError(
            "\n[pymrsf] OpenAI provider requires the openai package.\n"
            "  Install with: pip install pymrsf[openai]\n"
            "  Or use the local provider: Set PYMRSF_PROVIDER=local\n"
        )

    MODEL_VERSION = os.getenv("PYMRSF_MODEL_VERSION", "gpt-3.5-turbo")
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        raise ValueError(
            "\n[pymrsf] OpenAI provider requires OPENAI_API_KEY environment variable.\n"
            "  Set it in your .env file or export it:\n"
            "    export OPENAI_API_KEY='sk-...'\n"
            "  Or use the local provider: Set PYMRSF_PROVIDER=local\n"
        )
    
    _client = OpenAI(api_key=api_key)
    _logger.info("Using OpenAI provider: %s", MODEL_VERSION)
    _logger.info("Note: Advanced features (knowledge probing) require local provider.")

    def tokenize(text: str) -> list:
        try:
            import tiktoken
            enc = tiktoken.encoding_for_model(MODEL_VERSION)
            return enc.encode(text)
        except Exception as e:
            _logger.warning("tiktoken failed (%s), falling back to split()", e)
            return text.split()

    def detokenize(ids: list) -> str:
        """Convert token IDs back to string. Uses tiktoken if available."""
        try:
            import tiktoken
            enc = tiktoken.encoding_for_model(MODEL_VERSION)
            return enc.decode(ids)
        except Exception as e:
            _logger.warning("tiktoken decode failed (%s), falling back to str join", e)
            return " ".join(str(i) for i in ids)

    def _quantized_argmax(raw_logits) -> int:
        raise NotImplementedError(
            "\n[pymrsf] This feature requires the local provider (not available with OpenAI).\n"
            "  Advanced features require direct model access via llama-cpp-python.\n"
            "  Install with: pip install pymrsf[local]\n"
            "  And set: PYMRSF_PROVIDER=local\n"
        )

    quantized_argmax = _quantized_argmax

    def get_surprises(text: str) -> tuple:
        # The OpenAI Chat Completions API only returns logprobs for the tokens it
        # *generates*, not for the tokens of the input `text`. The previous
        # implementation requested max_tokens=1 and reported logprobs of that one
        # generated token — i.e. it measured a different string than `text`, which
        # is misleading. Until a true echo/scoring endpoint is wired up, report no
        # surprises (consistent with supports_true_surprises=False) instead of
        # fabricating a heatmap for the wrong tokens.
        _logger.warning(
            "get_surprises: input-token surprises are not available via the OpenAI "
            "Chat Completions API. Returning empty surprises. Use the local provider "
            "for token-level novelty detection."
        )
        tokens = tokenize(text)
        n = len(tokens)
        heatmap = [
            {"token": str(tok), "surprised": False, "position": i,
             "logprob": None, "prob": None}
            for i, tok in enumerate(tokens)
        ]
        return [], heatmap, n

    def compute_delta(text_or_ids) -> list:
        raise NotImplementedError(
            "\n[pymrsf] This feature requires the local provider (not available with OpenAI).\n"
            "  Advanced features require direct model access via llama-cpp-python.\n"
            "  Install with: pip install pymrsf[local]\n"
            "  And set: PYMRSF_PROVIDER=local\n"
        )

    class ModelSession:
        def __init__(self):
            raise NotImplementedError(
                "\n[pymrsf] ModelSession requires the local provider (not available with OpenAI).\n"
                "  This feature requires direct model access via llama-cpp-python.\n"
                "  Install with: pip install pymrsf[local]\n"
                "  And set: PYMRSF_PROVIDER=local\n"
            )

    def next_token_greedy(context_ids: list) -> int:
        raise NotImplementedError(
            "\n[pymrsf] This feature requires the local provider (not available with OpenAI).\n"
            "  Advanced features require direct model access via llama-cpp-python.\n"
            "  Install with: pip install pymrsf[local]\n"
            "  And set: PYMRSF_PROVIDER=local\n"
        )

    return {
        "tokenize": tokenize,
        "detokenize": detokenize,
        "quantized_argmax": _quantized_argmax,
        "get_surprises": get_surprises,
        "compute_delta": compute_delta,
        "ModelSession": ModelSession,
        "next_token_greedy": next_token_greedy,
        "lm": None,
    }


# ── Anthropic provider ─────────────────────────────────────────────────────────

def _load_anthropic_backend():
    """Dynamically load the Anthropic backend functions.
    
    Note: Anthropic API does not expose token logprobs, so novelty detection
    is limited. This provider is best used for embeddings and basic RAG scoring
    without novelty-based filtering.
    """
    try:
        from anthropic import Anthropic
    except ImportError:
        raise ImportError(
            "\n[pymrsf] Anthropic provider requires the anthropic package.\n"
            "  Install with: pip install pymrsf[anthropic]\n"
            "  Or use the local provider: Set PYMRSF_PROVIDER=local\n"
        )

    MODEL_VERSION = os.getenv("PYMRSF_MODEL_VERSION", "claude-3-5-sonnet-20241022")
    api_key = os.getenv("ANTHROPIC_API_KEY")
    
    if not api_key:
        raise ValueError(
            "\n[pymrsf] Anthropic provider requires ANTHROPIC_API_KEY environment variable.\n"
            "  Set it in your .env file or export it:\n"
            "    export ANTHROPIC_API_KEY='sk-ant-...'\n"
            "  Or use the local provider: Set PYMRSF_PROVIDER=local\n"
        )
    
    _client = Anthropic(api_key=api_key)
    _logger.info("Using Anthropic provider: %s", MODEL_VERSION)
    _logger.info("Anthropic does not expose logprobs — using relevance-only RAG scoring.")

    def tokenize(text: str) -> list:
        """Approximate tokenization using Claude's tokenizer or fallback."""
        try:
            import tiktoken
            # Use GPT-4 tokenizer as approximation for Claude
            enc = tiktoken.encoding_for_model("gpt-4")
            return enc.encode(text)
        except Exception:
            # Fallback: simple whitespace split
            return text.split()

    def detokenize(ids: list) -> str:
        """Convert token IDs back to string."""
        try:
            import tiktoken
            enc = tiktoken.encoding_for_model("gpt-4")
            return enc.decode(ids)
        except Exception:
            return " ".join(str(i) for i in ids)

    def _quantized_argmax(raw_logits) -> int:
        raise NotImplementedError(
            "\n[pymrsf] This feature requires the local provider (not available with Anthropic).\n"
            "  Anthropic API does not expose token logprobs.\n"
            "  Install with: pip install pymrsf[local]\n"
            "  And set: PYMRSF_PROVIDER=local\n"
        )

    def get_surprises(text: str) -> tuple:
        """
        Anthropic doesn't provide logprobs, so we can't detect surprises.
        Return empty surprises and basic token heatmap.
        """
        tokens = tokenize(text)
        n = len(tokens)
        surprises = []
        heatmap = []
        
        # Since we can't determine surprises, mark all as not surprised
        for i, tok_id in enumerate(tokens):
            tok_str = str(tok_id)  # Approximate
            heatmap.append({
                "token": tok_str,
                "surprised": False,
                "position": i,
                "logprob": None,
                "prob": None,
            })
        
        return surprises, heatmap, n

    def compute_delta(text_or_ids) -> list:
        raise NotImplementedError(
            "\n[pymrsf] This feature requires the local provider (not available with Anthropic).\n"
            "  Anthropic API does not expose token logprobs.\n"
            "  Install with: pip install pymrsf[local]\n"
            "  And set: PYMRSF_PROVIDER=local\n"
        )

    class ModelSession:
        def __init__(self):
            raise NotImplementedError(
                "\n[pymrsf] ModelSession requires the local provider (not available with Anthropic).\n"
                "  This feature requires direct model access via llama-cpp-python.\n"
                "  Install with: pip install pymrsf[local]\n"
                "  And set: PYMRSF_PROVIDER=local\n"
            )

    def next_token_greedy(context_ids: list) -> int:
        raise NotImplementedError(
            "\n[pymrsf] This feature requires the local provider (not available with Anthropic).\n"
            "  Anthropic API does not expose token logprobs.\n"
            "  Install with: pip install pymrsf[local]\n"
            "  And set: PYMRSF_PROVIDER=local\n"
        )

    return {
        "tokenize": tokenize,
        "detokenize": detokenize,
        "quantized_argmax": _quantized_argmax,
        "get_surprises": get_surprises,
        "compute_delta": compute_delta,
        "ModelSession": ModelSession,
        "next_token_greedy": next_token_greedy,
        "lm": None,
    }


# ── Backend router ─────────────────────────────────────────────────────────────

_backend = None


def _get_backend():
    global _backend
    if _backend is None:
        if PROVIDER == "local":
            _backend = _load_local_backend()
        elif PROVIDER == "openai":
            _backend = _load_openai_backend()
        elif PROVIDER == "anthropic":
            _backend = _load_anthropic_backend()
        else:
            raise ValueError(
                f"\n[pymrsf] Unknown provider: '{PROVIDER}'\n"
                f"  Valid options: local, openai, anthropic\n"
                f"  Set PYMRSF_PROVIDER in your .env file.\n"
            )
    return _backend


# ── Public API (lazy-loaded proxies) ───────────────────────────────────────────

def tokenize(text: str) -> list:
    """
    Convert text to token IDs.
    
    Multi-provider: Available with all providers.
    - Local: Uses llama-cpp-python tokenizer (exact)
    - OpenAI/Anthropic: Uses tiktoken approximation
    
    Returns:
        List of token IDs
    """
    return _get_backend()["tokenize"](text)


def detokenize(ids: list) -> str:
    """
    Convert token IDs back to text.
    
    Multi-provider: Available with all providers.
    - Local: Uses llama-cpp-python detokenizer (exact)
    - OpenAI/Anthropic: Uses tiktoken approximation
    
    Returns:
        Decoded text string
    """
    return _get_backend()["detokenize"](ids)


def quantized_argmax(raw_logits) -> int:
    """
    Get the argmax of logits with quantization.
    
    Local-only: Requires direct access to raw logits.
    
    Args:
        raw_logits: Raw logit array from model
        
    Returns:
        Token ID with highest logit value
        
    Raises:
        NotImplementedError: If called with non-local provider
    """
    return _get_backend()["quantized_argmax"](raw_logits)


def get_surprises(text: str) -> tuple:
    """
    Get token-level surprise information.
    
    Provider support varies:
    - Local: Full token-level exact surprises via argmax
    - OpenAI: Limited threshold-based surprises via API logprobs
    - Anthropic: Not supported (no logprobs available)
    
    Returns:
        (surprises, heatmap, token_count) tuple
    """
    return _get_backend()["get_surprises"](text)


def compute_delta(text_or_ids) -> list:
    """
    Compute delta (surprise positions and token IDs) for compression.
    
    Local-only: Requires exact token prediction via argmax.
    
    Args:
        text_or_ids: Either a string or list of token IDs
        
    Returns:
        List of (position, token_id) tuples for surprise tokens
        
    Raises:
        NotImplementedError: If called with non-local provider
    """
    return _get_backend()["compute_delta"](text_or_ids)


def next_token_greedy(context_ids: list) -> int:
    """
    Predict next token greedily (legacy O(n²) version).
    
    Local-only: Requires direct model access.
    Use ModelSession instead for O(n) performance.
    
    Args:
        context_ids: List of token IDs for context
        
    Returns:
        Predicted next token ID
        
    Raises:
        NotImplementedError: If called with non-local provider
    """
    return _get_backend()["next_token_greedy"](context_ids)


class ModelSession:
    """
    Stateful session for incremental token generation.
    
    Local-only: Requires KV cache access.
    
    Maintains model state for O(n) reconstruction instead of O(n²).
    Feed tokens one by one; predict next token from current state.
    
    Example:
        >>> session = ModelSession()
        >>> session.reset()
        >>> session.feed(token_id)
        >>> next_tok = session.predict_next()
    
    Raises:
        NotImplementedError: If instantiated with non-local provider
    """
    def __init__(self):
        self._session = _get_backend()["ModelSession"]()

    def reset(self):
        """Reset model state (clear KV cache)."""
        self._session.reset()

    def feed(self, token_id: int):
        """Feed a single token to update model state."""
        self._session.feed(token_id)

    def predict_next(self) -> int:
        """Return the greedy next token based on current state."""
        return self._session.predict_next()


lm = None  # not safe to expose directly anymore; use get_raw_lm() instead


# ── Provider capabilities ──────────────────────────────────────────────────────

def provider_capabilities() -> dict:
    """
    Returns a dictionary describing what features are available with the current provider.
    
    Use this to check feature availability at runtime before calling provider-specific functions.
    
    Returns:
        {
            "provider": str,              # "local", "openai", or "anthropic"
            "supports_logits": bool,      # Full logit access (quantized_argmax)
            "supports_probe": bool,       # Knowledge probing
            "supports_delta": bool,       # Delta compression
            "supports_sessions": bool,    # Stateful KV-cached generation
            "supports_true_surprises": bool,  # Token-level exact surprise detection
            "supports_embeddings": bool,  # Semantic embeddings
            "supports_tokenization": bool,  # Tokenization (may be approximate)
        }
    
    Example:
        >>> caps = provider_capabilities()
        >>> if caps["supports_probe"]:
        ...     result = probe("Hello world")
        >>> else:
        ...     print("Probing unavailable with this provider")
    """
    capabilities = {
        "provider": PROVIDER,
        "supports_tokenization": True,  # All providers (may be approximate)
        "supports_embeddings": True,    # All providers support embeddings
    }
    
    if PROVIDER == "local":
        capabilities.update({
            "supports_logits": True,
            "supports_probe": True,
            "supports_delta": True,
            "supports_sessions": True,
            "supports_true_surprises": True,
        })
    elif PROVIDER == "openai":
        capabilities.update({
            "supports_logits": False,      # No raw logits
            "supports_probe": False,       # Needs exact surprises
            "supports_delta": False,       # Needs exact surprises
            "supports_sessions": False,    # No KV cache access
            "supports_true_surprises": False,  # Only threshold-based via API
        })
    elif PROVIDER == "anthropic":
        capabilities.update({
            "supports_logits": False,
            "supports_probe": False,
            "supports_delta": False,
            "supports_sessions": False,
            "supports_true_surprises": False,
        })
    
    return capabilities


def get_backend():
    """
    Public accessor for the backend dictionary.
    Returns the loaded backend with all provider-specific functions.
    """
    return _get_backend()


def get_raw_lm():
    """
    Get direct access to the underlying language model object.
    Only available with local provider.
    
    Returns:
        The llama_cpp.Llama object for local provider, or None for API providers.
        
    Raises:
        NotImplementedError if called with a provider that doesn't support raw access.
    """
    backend = _get_backend()
    lm_obj = backend.get("lm")
    
    if lm_obj is None and PROVIDER != "local":
        raise NotImplementedError(
            f"\n[pymrsf] Raw model access not available with {PROVIDER} provider.\n"
            f"  This feature requires the local provider.\n"
            f"  Install with: pip install pymrsf[local]\n"
            f"  And set: PYMRSF_PROVIDER=local\n"
        )
    
    return lm_obj


# ── MODEL_VERSION (provider-aware) ─────────────────────────────────────────────

def _get_model_version() -> str:
    """Get the current model version string based on provider."""
    if PROVIDER == "local":
        return os.getenv("PYMRSF_MODEL_VERSION", "mistral-7b-q4km-v1")
    elif PROVIDER == "openai":
        return os.getenv("PYMRSF_MODEL_VERSION", "gpt-3.5-turbo")
    elif PROVIDER == "anthropic":
        return os.getenv("PYMRSF_MODEL_VERSION", "claude-3-5-sonnet-20241022")
    else:
        return "unknown"


MODEL_VERSION = _get_model_version()


def set_provider(name: str) -> None:
    """Switch providers at runtime.

    Resets cached model + backend state and updates PYMRSF_PROVIDER. Switching
    away from 'local' releases the GGUF model from memory on next use; switching
    to 'local' triggers a fresh model load on first use. Also syncs the global
    Config so get_config().provider reflects the change.

    Args:
        name: Provider name — "local", "openai", or "anthropic"
    """
    global PROVIDER, MODEL_VERSION, _lm, _lm_loaded, _backend
    PROVIDER = name.lower()
    _lm = None
    _lm_loaded = False
    _backend = None  # drop the cached backend so the new provider's fns load
    os.environ["PYMRSF_PROVIDER"] = PROVIDER
    MODEL_VERSION = _get_model_version()
    # Keep the central Config in sync (lazy import to avoid an import cycle).
    try:
        from pymrsf import _config
        _config.provider = PROVIDER
    except Exception as e:  # pragma: no cover - sync is best-effort
        _logger.debug("could not sync Config.provider: %s", e)
