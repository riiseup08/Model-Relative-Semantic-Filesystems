"""
pymrsf — multi-provider backend
Supports:
  - local    : any GGUF model via llama-cpp-python
  - openai   : GPT-3.5, GPT-4 via OpenAI logprobs API
  - anthropic: NOT supported (no logprobs API exposed)

Set provider in .env:
  PYMRSF_PROVIDER=local    # default
  PYMRSF_PROVIDER=openai
"""

import os
from dotenv import load_dotenv

load_dotenv()

PROVIDER        = os.getenv("PYMRSF_PROVIDER", "local").lower()
LOGIT_PRECISION = int(os.getenv("PYMRSF_LOGIT_PRECISION", "6"))

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
    if PROVIDER == "local":
        import numpy as np
        from llama_cpp import Llama

        GGUF_PATH     = os.getenv("PYMRSF_MODEL_PATH",    "./models/mistral-7b-v0.1.Q4_K_M.gguf")
        N_CTX         = int(os.getenv("PYMRSF_N_CTX",         "4096"))
        N_GPU_LAYERS  = int(os.getenv("PYMRSF_N_GPU_LAYERS",  "0"))
        N_THREADS     = int(os.getenv("PYMRSF_N_THREADS",     str(os.cpu_count())))

        if not os.path.exists(GGUF_PATH):
            raise FileNotFoundError(
                f"\n[pymrsf] Model not found: {GGUF_PATH}\n"
                f"  Set PYMRSF_MODEL_PATH in your .env"
            )

        print(f"[pymrsf] Loading local model: {GGUF_PATH}")
        _lm = Llama(
            model_path   = GGUF_PATH,
            n_ctx        = N_CTX,
            n_gpu_layers = N_GPU_LAYERS,
            logits_all   = True,
            verbose      = False,
            n_threads    = N_THREADS,
        )
        print(f"[pymrsf] Model loaded.\n")
        _lm_loaded = True
    elif PROVIDER == "openai":
        _lm_loaded = True  # No LLM to load, just mark as ready
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
        raise ImportError("[pymrsf] OpenAI provider requires: pip install openai")

    MODEL_VERSION = os.getenv("PYMRSF_MODEL_VERSION", "gpt-3.5-turbo")
    _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    print(f"[pymrsf] Using OpenAI provider: {MODEL_VERSION}\n")

    def tokenize(text: str) -> list:
        try:
            import tiktoken
            enc = tiktoken.encoding_for_model(MODEL_VERSION)
            return enc.encode(text)
        except Exception as e:
            print(f"[pymrsf] Warning: tiktoken failed ({e}), falling back to split()")
            return text.split()

    def detokenize(ids: list) -> str:
        """Convert token IDs back to string. Uses tiktoken if available."""
        try:
            import tiktoken
            enc = tiktoken.encoding_for_model(MODEL_VERSION)
            return enc.decode(ids)
        except Exception as e:
            print(f"[pymrsf] Warning: tiktoken decode failed ({e}), falling back to str join")
            return " ".join(str(i) for i in ids)

    def _quantized_argmax(raw_logits) -> int:
        raise NotImplementedError(
            "[pymrsf] quantized_argmax is not available for OpenAI provider.\n"
            "  mrsf_write / mrsf_read require a local model."
        )

    def get_surprises(text: str) -> tuple:
        import math
        SURPRISE_THRESHOLD = float(os.getenv("PYMRSF_SURPRISE_THRESHOLD", "-1.0"))
        response = _client.chat.completions.create(
            model=MODEL_VERSION,
            messages=[{"role": "user", "content": text}],
            logprobs=True,
            max_tokens=1,
        )
        token_logprobs = response.choices[0].logprobs.content or []
        surprises = []
        heatmap = []
        for i, token_info in enumerate(token_logprobs):
            tok = token_info.token
            logprob = token_info.logprob
            surprised = (logprob is not None and logprob < SURPRISE_THRESHOLD)
            if surprised:
                surprises.append((i, tok))
            heatmap.append({
                "token": tok,
                "surprised": surprised,
                "position": i,
                "logprob": round(logprob, 4) if logprob else None,
                "prob": round(math.exp(logprob), 4) if logprob else 0.0,
            })
        n = len(token_logprobs)
        return surprises, heatmap, n

    def compute_delta(text_or_ids) -> list:
        raise NotImplementedError(
            "[pymrsf] compute_delta is not available for OpenAI provider.\n"
            "  mrsf_write requires a local model."
        )

    class ModelSession:
        def __init__(self):
            raise NotImplementedError(
                "[pymrsf] ModelSession requires local provider.\n"
                "  mrsf_write / mrsf_read require a local model."
            )

    def next_token_greedy(context_ids: list) -> int:
        raise NotImplementedError(
            "[pymrsf] next_token_greedy is not available for OpenAI provider.\n"
            "  mrsf_write / mrsf_read require a local model."
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
            raise NotImplementedError(
                "\n[pymrsf] Anthropic / Claude is not supported.\n"
                "  Anthropic's API does not expose token logprobs — \n"
                "  which are required for pymrsf's compression calculation.\n"
                "  Supported providers: local, openai\n"
                "  See: https://docs.anthropic.com/en/api/messages"
            )
        else:
            raise ValueError(
                f"\n[pymrsf] Unknown provider: '{PROVIDER}'\n"
                f"  Valid options: local, openai\n"
                f"  Set PYMRSF_PROVIDER in your .env file."
            )
    return _backend


# ── Public API (lazy-loaded proxies) ───────────────────────────────────────────

def tokenize(text: str) -> list:
    return _get_backend()["tokenize"](text)


def detokenize(ids: list) -> str:
    return _get_backend()["detokenize"](ids)


def quantized_argmax(raw_logits) -> int:
    return _get_backend()["quantized_argmax"](raw_logits)


def get_surprises(text: str) -> tuple:
    return _get_backend()["get_surprises"](text)


def compute_delta(text_or_ids) -> list:
    return _get_backend()["compute_delta"](text_or_ids)


def next_token_greedy(context_ids: list) -> int:
    return _get_backend()["next_token_greedy"](context_ids)


class ModelSession:
    def __init__(self):
        self._session = _get_backend()["ModelSession"]()

    def reset(self):
        self._session.reset()

    def feed(self, token_id: int):
        self._session.feed(token_id)

    def predict_next(self) -> int:
        return self._session.predict_next()


lm = None  # not safe to expose directly anymore; use get_surprises() / compute_delta()

# Public constants
MODEL_VERSION = os.getenv("PYMRSF_MODEL_VERSION", "mistral-7b-q4km-v1")
