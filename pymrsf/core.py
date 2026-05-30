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

from dotenv import load_dotenv

_logger = logging.getLogger("pymrsf.core")

load_dotenv()


def _cfg():
    """Return the current Config, or None if pymrsf isn't fully initialised."""
    try:
        import pymrsf
        return pymrsf._config
    except (ImportError, AttributeError):
        return None


def _provider() -> str:
    cfg = _cfg()
    return (cfg.provider if cfg else os.getenv("PYMRSF_PROVIDER", "local")).lower()


def _logit_precision() -> int:
    cfg = _cfg()
    return cfg.logit_precision if cfg else int(os.getenv("PYMRSF_LOGIT_PRECISION", "6"))


# ── Lazy model loading ────────────────────────────────────────────────────────
# The LLM model is NOT loaded at import time. It's loaded on first use.
# This avoids loading a 4GB+ model when you only import for RAG scoring.

_lm = None
_lm_loaded = False


def _ensure_model():
    """Lazy-load the LLM model on first actual use."""
    global _lm, _lm_loaded
    if _lm_loaded:
        return
    provider = _provider()
    if provider == "local":
        try:
            import numpy as np  # noqa: F401  — availability check
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "\n[pymrsf] Local provider requires llama-cpp-python.\n"
                "  Install with: pip install pymrsf[local]\n"
                "  Or use a lightweight API provider instead:\n"
                "    Set PYMRSF_PROVIDER=openai (requires OpenAI API key)\n"
                "    Set PYMRSF_PROVIDER=anthropic (requires Anthropic API key)\n"
            )

        cfg = _cfg()
        GGUF_PATH    = cfg.model_path if cfg else os.getenv("PYMRSF_MODEL_PATH", "./models/mistral-7b-v0.1.Q4_K_M.gguf")
        N_CTX        = cfg.n_ctx if cfg else int(os.getenv("PYMRSF_N_CTX", "4096"))
        N_GPU_LAYERS = cfg.n_gpu_layers if cfg else int(os.getenv("PYMRSF_N_GPU_LAYERS", "0"))
        N_THREADS    = cfg.n_threads if cfg else int(os.getenv("PYMRSF_N_THREADS", str(os.cpu_count() or 4)))

        if not os.path.exists(GGUF_PATH):
            raise FileNotFoundError(
                f"\n[pymrsf] Model not found: {GGUF_PATH}\n"
                f"  Set PYMRSF_MODEL_PATH in your .env or call pymrsf.configure(model_path=...)"
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
    elif provider == "openai":
        _lm_loaded = True  # No LLM to load, just mark as ready
    elif provider == "anthropic":
        _lm_loaded = True  # No LLM to load — Anthropic doesn't expose logprobs
    else:
        raise ValueError(f"[pymrsf] Unknown provider: '{provider}'")


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
        q = np.round(np.array(raw_logits, dtype=np.float64), decimals=_logit_precision())
        return int(np.argmax(q))

    quantized_argmax = _quantized_argmax

    def get_surprises(text: str) -> tuple:
        """Returns (surprises, heatmap, token_count)"""
        token_ids = tokenize(text)
        n = len(token_ids)
        _lm.reset()
        _lm.eval(token_ids)

        surprises = []
        heatmap = []

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
        _lm.reset()
        _lm.eval(ids)
        delta = []
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

    cfg = _cfg()
    model_version = (cfg.model_version if cfg else os.getenv("PYMRSF_MODEL_VERSION", "")) or "gpt-3.5-turbo"
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError(
            "\n[pymrsf] OpenAI provider requires OPENAI_API_KEY environment variable.\n"
            "  Set it in your .env file or export it:\n"
            "    export OPENAI_API_KEY='sk-...'\n"
            "  Or use the local provider: Set PYMRSF_PROVIDER=local\n"
        )

    _client = OpenAI(api_key=api_key)
    _logger.info("Using OpenAI provider: %s", model_version)
    _logger.info("Note: Advanced features (knowledge probing) require local provider.")

    def tokenize(text: str) -> list:
        try:
            import tiktoken
            enc = tiktoken.encoding_for_model(model_version)
            return enc.encode(text)
        except Exception as e:
            _logger.warning("tiktoken failed (%s), falling back to split()", e)
            return text.split()

    def detokenize(ids: list) -> str:
        """Convert token IDs back to string. Uses tiktoken if available."""
        try:
            import tiktoken
            enc = tiktoken.encoding_for_model(model_version)
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


    def get_surprises(text: str) -> tuple:
        """
        Token-level surprise detection via OpenAI legacy completions endpoint.

        Uses echo=True + logprobs on a completion-model endpoint to get
        per-position logprobs for the entire input prompt. This requires
        a model that supports the legacy completions API
        (e.g. gpt-3.5-turbo-instruct).

        Raises:
            NotImplementedError: If the model doesn't support echo logprobs.
        """
        import math
        _COMPLETION_MODELS = {"gpt-3.5-turbo-instruct", "davinci-002", "babbage-002"}

        def _supports_echo_logprobs(m: str) -> bool:
            return m in _COMPLETION_MODELS or m.endswith("-instruct")

        if not _supports_echo_logprobs(model_version):
            raise NotImplementedError(
                f"\n[pymrsf] OpenAI model '{model_version}' does not support "
                "input-prompt logprobs (the Chat Completions API only returns "
                "logprobs of generated tokens, not the prompt).\n"
                "  Options:\n"
                "    1. pymrsf.configure(model_version='gpt-3.5-turbo-instruct')\n"
                "    2. Switch to the local provider: pymrsf.configure(provider='local')\n"
                "    3. Use score_chunk() / filter_chunks() which fall back to "
                "relevance-only scoring without surprises.\n"
            )

        threshold = (cfg.surprise_threshold if cfg
                     else float(os.getenv("PYMRSF_SURPRISE_THRESHOLD", "-1.0")))

        response = _client.completions.create(
            model=model_version, prompt=text,
            max_tokens=0, echo=True, logprobs=5,
        )

        lp = response.choices[0].logprobs
        tokens = lp.tokens
        token_logprobs = lp.token_logprobs
        top_logprobs = lp.top_logprobs

        n = len(tokens)
        surprises = []
        heatmap = []

        for i in range(n):
            logprob = token_logprobs[i]
            if i == 0 or logprob is None:
                heatmap.append({
                    "token": tokens[i],
                    "surprised": False,
                    "position": i,
                    "logprob": None,
                    "prob": None,
                })
                continue

            # Two signals: argmax surprise and threshold surprise
            argmax_surprise = False
            if top_logprobs[i]:
                top_token = max(top_logprobs[i].items(), key=lambda kv: kv[1])[0]
                argmax_surprise = top_token != tokens[i]
            threshold_surprise = logprob < threshold
            surprised = argmax_surprise or threshold_surprise

            if surprised:
                surprises.append((i, tokens[i]))
            heatmap.append({
                "token": tokens[i],
                "surprised": surprised,
                "position": i,
                "logprob": round(logprob, 4),
                "prob": round(math.exp(logprob), 4),
            })

        return surprises, heatmap, n

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

    cfg = _cfg()
    model_version = (cfg.model_version if cfg else os.getenv("PYMRSF_MODEL_VERSION", "")) or "claude-3-5-sonnet-20241022"
    api_key = os.getenv("ANTHROPIC_API_KEY")

    if not api_key:
        raise ValueError(
            "\n[pymrsf] Anthropic provider requires ANTHROPIC_API_KEY environment variable.\n"
            "  Set it in your .env file or export it:\n"
            "    export ANTHROPIC_API_KEY='sk-ant-...'\n"
            "  Or use the local provider: Set PYMRSF_PROVIDER=local\n"
        )

    _client = Anthropic(api_key=api_key)
    _logger.info("Using Anthropic provider: %s", model_version)
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
        Return empty surprises and basic token heatmap with decoded strings.
        """
        tokens = tokenize(text)
        n = len(tokens)
        surprises = []
        heatmap = []

        # Since we can't determine surprises, mark all as not surprised
        for i, tok_id in enumerate(tokens):
            tok_str = detokenize([tok_id]).strip() or f"<{tok_id}>"
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
        provider = _provider()
        if provider == "local":
            _backend = _load_local_backend()
        elif provider == "openai":
            _backend = _load_openai_backend()
        elif provider == "anthropic":
            _backend = _load_anthropic_backend()
        else:
            raise ValueError(
                f"\n[pymrsf] Unknown provider: '{provider}'\n"
                f"  Valid options: local, openai, anthropic\n"
                f"  Call pymrsf.configure(provider=...) or set PYMRSF_PROVIDER in your .env file.\n"
            )
    return _backend


# ── Public API (lazy-loaded proxies) ───────────────────────────────────────────

def tokenize(text: str) -> list:
    """Convert text to token IDs.

    Available with all providers. Local provider uses the exact model tokenizer;
    OpenAI and Anthropic providers use tiktoken as an approximation.

    Args:
        text: Text to tokenize

    Returns:
        List of token IDs

    Example:
        >>> tokenize("Hello world")
        [1, 22557, 1526]
    """
    return _get_backend()["tokenize"](text)


def detokenize(ids: list) -> str:
    """Convert token IDs back to text.

    Available with all providers. Local provider uses the exact model detokenizer;
    OpenAI and Anthropic providers use tiktoken as an approximation.

    Args:
        ids: List of token IDs to decode

    Returns:
        Decoded text string

    Example:
        >>> detokenize([22557, 1526])
        'Hello world'
    """
    return _get_backend()["detokenize"](ids)


def quantized_argmax(raw_logits) -> int:
    """Get the token ID with the highest quantized logit value.

    Local-only: requires direct access to raw logits. The quantization precision
    is controlled by PYMRSF_LOGIT_PRECISION (default 6 decimal places).

    Args:
        raw_logits: Raw logit array from model

    Returns:
        Token ID with highest logit value after quantization

    Raises:
        NotImplementedError: If called with non-local provider

    Example:
        >>> quantized_argmax(np.array([0.1, 0.9, 0.3]))
        1
    """
    return _get_backend()["quantized_argmax"](raw_logits)


def get_surprises(text: str) -> tuple:
    """Get token-level surprise information.

    Provider support varies:
    - Local: Full token-level exact surprises via argmax
    - OpenAI: Limited threshold-based surprises via API logprobs
    - Anthropic: Not supported (no logprobs available)

    Args:
        text: Text to analyze for surprise tokens

    Returns:
        (surprises, heatmap, token_count) tuple

    Example:
        >>> surprises, heatmap, n = get_surprises("The Eiffel Tower is tall.")
        >>> len(surprises)
        0
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
    """Predict the next token using greedy decoding (legacy O(n²)).

    Local-only: requires direct model access. For O(n) performance use
    ModelSession instead — this function resets the model for every call.

    Args:
        context_ids: List of token IDs forming the context

    Returns:
        Predicted next token ID

    Raises:
        NotImplementedError: If called with non-local provider

    Example:
        >>> next_token_greedy([1, 22557])
        1526
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
    provider = _provider()
    capabilities = {
        "provider": provider,
        "supports_tokenization": True,  # All providers (may be approximate)
        "supports_embeddings": True,    # All providers support embeddings
    }

    if provider == "local":
        capabilities.update({
            "supports_logits": True,
            "supports_probe": True,
            "supports_delta": True,
            "supports_sessions": True,
            "supports_true_surprises": True,
        })
    elif provider == "openai":
        capabilities.update({
            "supports_logits": False,      # No raw logits
            "supports_probe": False,       # Needs exact surprises
            "supports_delta": False,       # Needs exact surprises
            "supports_sessions": False,    # No KV cache access
            "supports_true_surprises": False,  # Only threshold-based via API
        })
    elif provider == "anthropic":
        capabilities.update({
            "supports_logits": False,
            "supports_probe": False,
            "supports_delta": False,
            "supports_sessions": False,
            "supports_true_surprises": False,
        })

    return capabilities


def get_backend():
    """Get the current provider backend dictionary.

    Returns a dict of provider-specific functions (tokenize, detokenize,
    quantized_argmax, etc.) and the raw lm object. The backend is loaded
    lazily on first call and cached thereafter.

    Returns:
        dict with keys: tokenize, detokenize, quantized_argmax, get_surprises,
        compute_delta, ModelSession, next_token_greedy, lm

    Example:
        >>> backend = get_backend()
        >>> backend.keys()
        dict_keys(['tokenize', 'detokenize', ...])
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

    if lm_obj is None and _provider() != "local":
        raise NotImplementedError(
            f"\n[pymrsf] Raw model access not available with {_provider()} provider.\n"
            f"  This feature requires the local provider.\n"
            f"  Install with: pip install pymrsf[local]\n"
            f"  And set: PYMRSF_PROVIDER=local\n"
        )

    return lm_obj


# ── MODEL_VERSION (provider-aware) ─────────────────────────────────────────────

_MODEL_DEFAULTS = {
    "local": "mistral-7b-q4km-v1",
    "openai": "gpt-3.5-turbo",
    "anthropic": "claude-3-5-sonnet-20241022",
}


def _get_model_version() -> str:
    """Get the current model version string (live from Config, env, or provider default)."""
    cfg = _cfg()
    explicit = (cfg.model_version if cfg else os.getenv("PYMRSF_MODEL_VERSION", ""))
    if explicit:
        return explicit
    return _MODEL_DEFAULTS.get(_provider(), "unknown")


def get_provider() -> str:
    """Return the current provider name (live, not captured at import time)."""
    return _provider()


def get_model_version() -> str:
    """Return the current model version string (live, not captured at import time)."""
    return _get_model_version()


def set_provider(name: str) -> None:
    """Switch providers at runtime.

    Delegates to pymrsf.configure() which invalidates model state and
    syncs the env var. This ensures both code paths use identical reset logic.

    Args:
        name: Provider name — "local", "openai", or "anthropic"

    Example:
        >>> set_provider("openai")
    """
    import pymrsf
    pymrsf.configure(provider=name.lower())


# ── PEP 562 __getattr__ for backward-compat module-level constants ──────────

def __getattr__(name):
    if name == "PROVIDER":
        return _provider()
    if name == "LOGIT_PRECISION":
        return _logit_precision()
    if name == "MODEL_VERSION":
        return _get_model_version()
    raise AttributeError(f"module 'pymrsf.core' has no attribute {name!r}")
