"""
pymrsf.chunker — Surprise-guided auto-chunking

Splits text at knowledge boundaries using the model's own signal:

  - **Local provider**: uses token-level logprobs (surprise) to find
    natural topic transitions.
  - **API providers** (OpenAI, Anthropic): uses embedding cosine similarity
    between sliding windows to detect topic shifts.
  - **Fallback**: sentence-based splitting when neither is available.
"""

import logging
import re

import numpy as np

from .core import detokenize, get_surprises, provider_capabilities, tokenize

_logger = logging.getLogger("pymrsf.chunker")


# ── Embedding-based topic boundary detection (API providers) ───────────────


def _embed_boundaries(text: str) -> list[int]:
    """Find topic boundaries using embedding similarity between sliding windows.

    For API providers that don't expose token logprobs, we detect topic
    transitions by measuring cosine similarity between adjacent embedding
    windows.  A sharp drop in similarity indicates a topic boundary.

    Returns list of character offsets where boundaries should be placed.
    """
    from .embeddings import embed

    WINDOW_CHARS = 300
    STRIDE = 100
    SIM_THRESHOLD = 0.70

    if len(text) <= WINDOW_CHARS:
        return []

    # Build sliding windows
    windows: list[str] = []
    offsets: list[int] = []
    pos = 0
    while pos < len(text):
        chunk = text[pos : pos + WINDOW_CHARS]
        if len(chunk) < WINDOW_CHARS // 2:
            break  # skip tail shorter than half a window
        windows.append(chunk)
        offsets.append(pos)
        pos += STRIDE

    if len(windows) < 2:
        return []

    # Embed each window (embed() accepts a single string)
    try:
        vecs = np.array([embed(w) for w in windows], dtype=np.float32)
    except Exception:
        return []

    if len(vecs) < 2:
        return []

    # Normalize
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    vecs = vecs / norms

    # Find drops in similarity between adjacent windows
    # sims[i] = similarity between window[i] and window[i+1]
    sims = np.sum(vecs[:-1] * vecs[1:], axis=1)
    boundary_offsets: list[int] = []
    for i in range(1, len(sims)):
        if sims[i - 1] - sims[i] > (1.0 - SIM_THRESHOLD):
            # Place boundary midway between the two windows
            mid = (offsets[i] + offsets[i + 1]) // 2 if i + 1 < len(offsets) else offsets[i]
            boundary_offsets.append(mid)

    return sorted(set(boundary_offsets))


# ── Logprob-based topic boundary detection (local provider) ────────────────


def _logprob_boundaries(
    text: str,
    surprise_drop_threshold: float,
    window: int,
) -> list[int]:
    """Find topic boundaries using token-level logprob surprise.

    For the local provider with raw logit access.  Returns character
    offsets where boundaries should be placed.
    """
    surprises_data = get_surprises(text)

    if not surprises_data:
        return []

    # Normalise to list of (position, token_str, is_surprised)
    def _parse(item):
        if isinstance(item, dict):
            return item.get("position", 0), item.get("token", ""), bool(item.get("surprised", False))
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            return item[0], item[1], True
        return 0, "", False

    parsed = [_parse(s) for s in surprises_data]

    # Build binary surprise signal (1=surprised, 0=not) indexed by token position
    max_pos = max(p for p, _, _ in parsed) + 1 if parsed else 1
    signal = [0.0] * max_pos
    for pos, _, is_surprised in parsed:
        if 0 <= pos < max_pos:
            signal[pos] = 1.0 if is_surprised else 0.0

    # Rolling average
    rolling = []
    for i in range(max_pos):
        start = max(0, i - window + 1)
        rolling.append(sum(signal[start:i + 1]) / (i - start + 1))

    # Find boundary positions: rolling drops by >= surprise_drop_threshold from local peak
    boundary_positions: list[int] = []
    peak = rolling[0] if rolling else 0.0
    for i in range(1, len(rolling)):
        if rolling[i] > peak:
            peak = rolling[i]
        elif peak > 0 and (peak - rolling[i]) / peak >= surprise_drop_threshold:
            boundary_positions.append(i)
            peak = rolling[i]

    # Map token positions to character offsets
    try:
        token_ids = tokenize(text)
    except Exception:
        return []

    cum_len = 0
    cum_chars: list[int] = [0]
    for tid in token_ids:
        try:
            cum_len += len(detokenize([tid]))
        except Exception:
            pass
        cum_chars.append(cum_len)

    char_boundaries: list[int] = []
    for bp in boundary_positions:
        if bp < len(cum_chars):
            char_boundaries.append(cum_chars[bp])

    return char_boundaries


# ── Sentence-based fallback ────────────────────────────────────────────────


def _sentence_fallback(text: str, max_chunk_len: int) -> list[str]:
    """Simple sentence-based chunking used as fallback."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks, current = [], []
    current_len = 0
    for sent in sentences:
        if current_len + len(sent) > max_chunk_len and current:
            chunks.append(" ".join(current))
            current, current_len = [], 0
        current.append(sent)
        current_len += len(sent) + 1
    if current:
        chunks.append(" ".join(current))
    return [c for c in chunks if c.strip()]


# ── Post-processing helpers ────────────────────────────────────────────────


def _split_at_boundaries(text: str, char_boundaries: list[int]) -> list[str]:
    if not char_boundaries:
        return [text] if text.strip() else []

    raw_chunks: list[str] = []
    prev = 0
    for cb in sorted(set(char_boundaries)):
        raw_chunks.append(text[prev:cb])
        prev = cb
    raw_chunks.append(text[prev:])
    return [c for c in raw_chunks if c.strip()]


def _merge_short_chunks(chunks: list[str], min_chunk_len: int) -> list[str]:
    merged: list[str] = []
    buffer = ""
    for chunk in chunks:
        if len(buffer) + len(chunk) < min_chunk_len:
            buffer += (" " if buffer else "") + chunk.strip()
        else:
            if buffer:
                merged.append(buffer.strip())
            buffer = chunk.strip()
    if buffer:
        merged.append(buffer.strip())
    return merged


def _force_split_long(chunks: list[str], max_chunk_len: int) -> list[str]:
    final: list[str] = []
    for chunk in chunks:
        if len(chunk) <= max_chunk_len:
            final.append(chunk)
        else:
            final.extend(_sentence_fallback(chunk, max_chunk_len))
    return final


# ── Main public API ────────────────────────────────────────────────────────


def smart_chunk(
    text: str,
    min_chunk_len: int = 100,
    max_chunk_len: int = 1000,
    surprise_drop_threshold: float = 0.4,
    window: int = 5,
) -> list[str]:
    """Split text into semantically coherent chunks at knowledge boundaries.

    Boundary detection strategy (chosen automatically based on provider):
      - **Local provider**: token-level logprob surprise (most precise)
      - **API providers** (OpenAI, Anthropic): embedding cosine similarity
        between sliding windows (good topic shift detection)
      - **Fallback**: sentence-based splitting (no embeddings available)

    Args:
        text                   : Document text to chunk
        min_chunk_len          : Minimum characters per chunk (merge shorter ones)
        max_chunk_len          : Maximum characters per chunk (force-split longer ones)
        surprise_drop_threshold: Logprob-only — fractional drop in rolling surprise
                                 to trigger a boundary (0.4 = 40% drop)
        window                 : Logprob-only — rolling average window in tokens

    Returns:
        List of text chunk strings
    """
    if not text or not text.strip():
        return []

    caps = provider_capabilities()
    char_boundaries: list[int] = []

    if caps.get("supports_logits", False):
        # Local provider: logprob-based detection
        try:
            char_boundaries = _logprob_boundaries(text, surprise_drop_threshold, window)
        except Exception as e:
            _logger.warning("logprob chunking failed (%s) — trying embedding fallback.", e)

    if not char_boundaries and caps.get("supports_embeddings", False):
        # API provider or logprob failed: embedding-based detection
        try:
            char_boundaries = _embed_boundaries(text)
        except Exception as e:
            _logger.warning("embedding chunking failed (%s) — falling back to sentence splitting.", e)

    if not char_boundaries:
        _logger.info("No surprise/embedding signal — falling back to sentence chunking.")
        return _sentence_fallback(text, max_chunk_len)

    # Split at detected boundaries
    raw_chunks = _split_at_boundaries(text, char_boundaries)

    # Post-process
    chunks = _merge_short_chunks(raw_chunks, min_chunk_len)
    chunks = _force_split_long(chunks, max_chunk_len)

    return [c for c in chunks if c.strip()] or [text]
