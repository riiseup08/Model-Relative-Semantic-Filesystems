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


# ── Provider: local (llama-cpp) ───────────────────────────────────────────────

if PROVIDER == "local":
    import numpy as np
    from llama_cpp import Llama

    GGUF_PATH     = os.getenv("PYMRSF_MODEL_PATH",    "./models/mistral-7b-v0.1.Q4_K_M.gguf")
    MODEL_VERSION = os.getenv("PYMRSF_MODEL_VERSION", "mistral-7b-q4km-v1")
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
    print(f"[pymrsf] Model loaded: {MODEL_VERSION}\n")

    def tokenize(text: str) -> list:
        return _lm.tokenize(text.encode("utf-8"), add_bos=True)

    def detokenize(ids: list) -> str:
        """Convert token IDs back to string. Preserves all spaces."""
        return _lm.detokenize(ids).decode("utf-8", errors="replace")

    def _quantized_argmax(raw_logits) -> int:
        q = np.round(np.array(raw_logits, dtype=np.float64), decimals=LOGIT_PRECISION)
        return int(np.argmax(q))

    quantized_argmax = _quantized_argmax  # public alias

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

    # --- Stateful session for O(n) reconstruction ---
    class ModelSession:
        """
        Maintains a single model state for incremental generation.
        Feed tokens one by one; predict next token from current state.
        """
        def __init__(self):
            self.lm = _lm
            self.reset()

        def reset(self):
            """Reset model state (clear KV cache)."""
            self.lm.reset()
            self._last_logits = None

        def feed(self, token_id: int):
            """Feed a single token to the model and update internal logits."""
            self.lm.eval([token_id])
            # scores is a list where each element corresponds to logits after that input token
            # When we eval([tok]), scores has length 1, and scores[0] has the logits
            # But actually, scores contains logits for ALL positions evaluated so far
            # So we want the LAST one
            if len(self.lm.scores) == 0:
                self._last_logits = None
                print(f"[WARN] scores is empty after feeding {token_id}")
            else:
                self._last_logits = self.lm.scores[-1]

        def predict_next(self) -> int:
            """Return the greedy next token based on current state."""
            if self._last_logits is None:
                raise RuntimeError("No logits available. Call feed() first.")
            return _quantized_argmax(np.array(self._last_logits))

    # Keep old name for compatibility (though not recommended)
    def next_token_greedy(context_ids: list) -> int:
        """Legacy O(n²) version – use ModelSession instead."""
        _lm.reset()
        _lm.eval(context_ids)
        return _quantized_argmax(np.array(_lm.scores[len(context_ids) - 1]))

    lm = _lm  # exposed for direct access if needed


# ── Provider: openai ──────────────────────────────────────────────────────────

elif PROVIDER == "openai":
    try:
        from openai import OpenAI  # type: ignore
    except ImportError:
        raise ImportError("[pymrsf] OpenAI provider requires: pip install openai")

    MODEL_VERSION = os.getenv("PYMRSF_MODEL_VERSION", "gpt-3.5-turbo")
    _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    print(f"[pymrsf] Using OpenAI provider: {MODEL_VERSION}\n")

    def tokenize(text: str) -> list:
        try:
            import tiktoken  # type: ignore
            enc = tiktoken.encoding_for_model(MODEL_VERSION)
            return enc.encode(text)
        except Exception as e:
            print(f"[pymrsf] Warning: tiktoken failed ({e}), falling back to split()")
            return text.split()

    def detokenize(ids: list) -> str:
        """Convert token IDs back to string. Uses tiktoken if available."""
        try:
            import tiktoken  # type: ignore
            enc = tiktoken.encoding_for_model(MODEL_VERSION)
            return enc.decode(ids)
        except Exception as e:
            print(f"[pymrsf] Warning: tiktoken decode failed ({e}), falling back to str join")
            return " ".join(str(i) for i in ids)

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

    def next_token_greedy(context_ids: list) -> int:
        raise NotImplementedError(
            "[pymrsf] next_token_greedy is not available for OpenAI provider.\n"
            "  mrsf_write / mrsf_read require a local model."
        )

    class ModelSession:
        def __init__(self):
            raise NotImplementedError(
                "[pymrsf] ModelSession requires local provider.\n"
                "  mrsf_write / mrsf_read require a local model."
            )

    lm = None


# ── Provider: anthropic ───────────────────────────────────────────────────────

elif PROVIDER == "anthropic":
    raise NotImplementedError(
        "\n[pymrsf] Anthropic / Claude is not supported.\n"
        "  Anthropic's API does not expose token logprobs — \n"
        "  which are required for pymrsf's compression calculation.\n"
        "  Supported providers: local, openai\n"
        "  See: https://docs.anthropic.com/en/api/messages"
    )


# ── Unknown provider ──────────────────────────────────────────────────────────

else:
    raise ValueError(
        f"\n[pymrsf] Unknown provider: '{PROVIDER}'\n"
        f"  Valid options: local, openai\n"
        f"  Set PYMRSF_PROVIDER in your .env file."
    )